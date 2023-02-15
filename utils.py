"""
This script is intended to wake up every 30 min or so (eg via cron),
it checks for any new arxiv papers via the arxiv API and stashes
them into a sqlite database.
"""

import sys
import time
import random
import logging
import argparse
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from aslite.db import get_papers_db, save_features
from aslite.arxiv import get_response, parse_response
from aslite.db import get_papers_db, get_metas_db


class Arxiv:

    def __init__(self, num, start, break_after):
        self.num = num
        self.start_index = start
        self.break_index = break_after
        self.tags = ['stat.ML', 'stat.ME', 'stat.TH', 'eess.IV', 'eess.SP', 'eess.AS', 'quant-ph', 'physics.optics',
                     'physics.data-an', 'physics.class-ph', 'physics.app-ph', 'physics.atm-clus', 'physics.comp-ph',
                     'physics.plasm-ph', 'cond-mat.stat-mech', 'cond-mat.supr-con', 'cs.CV', 'cs.LG', 'cs.CL', 'cs.AI',
                     'cs.NE', 'cs.RO']
        self.q = None

    def format_api_query(self):
        formatted_tags = []

        for tag in self.tags:
            formatted_tags.append(f'cat:{tag}')

        self.q = "+OR+".join(formatted_tags)

    def fetch_papers(self):

        pdb = get_papers_db(flag='c')
        mdb = get_metas_db(flag='c')
        prevn = len(pdb)

        def store(p):
            pdb[p['_id']] = p
            mdb[p['_id']] = {'_time': p['_time']}

        # fetch the latest papers
        total_updated = 0
        zero_updates_in_a_row = 0
        for k in range(self.start_index, self.start_index + self.num, 100):

            print('querying arxiv api for query at start_index %d' % k)

            # attempt to fetch a batch of papers from arxiv api
            ntried = 0
            while True:
                try:
                    resp = get_response(search_query=self.q, start_index=k)
                    papers = parse_response(resp)
                    time.sleep(0.5)
                    if len(papers) == 100:
                        break  # otherwise we have to try again
                except Exception as e:
                    print(e)
                    print("will try again in a bit...")
                    ntried += 1
                    if ntried > 1000:
                        print("ok we tried 1,000 times, something is srsly wrong. exitting.")
                    time.sleep(2 + random.uniform(0, 4))

            # process the batch of retrieved papers
            nhad, nnew, nreplace = 0, 0, 0
            for p in papers:
                pid = p['_id']
                if pid in pdb:
                    if p['_time'] > pdb[pid]['_time']:
                        # replace, this one is newer
                        store(p)
                        nreplace += 1
                    else:
                        # we already had this paper, nothing to do
                        nhad += 1
                else:
                    # new, simple store into database
                    store(p)
                    nnew += 1
            prevn = len(pdb)
            total_updated += nreplace + nnew

            # some diagnostic information on how things are coming along
            print(papers[0]['_time_str'])
            print("k=%d, out of %d: had %d, replaced %d, new %d. now have: %d" %
                         (k, len(papers), nhad, nreplace, nnew, prevn))

            # early termination criteria
            if nnew == 0:
                zero_updates_in_a_row += 1
                if 0 < self.break_index <= zero_updates_in_a_row:
                    print("breaking out early, no new papers %d times in a row" % (self.break_index,))
                    break
                elif k == 0:
                    print("our very first call for the latest there were no new papers, exitting")
                    break
            else:
                zero_updates_in_a_row = 0

            # zzz
            time.sleep(1 + random.uniform(0, 3))


class Compute:

    def __init__(self, num, min_df, max_df, max_docs):
        self.num = num
        self.min_df = min_df
        self.max_df = max_df
        self.max_docs = max_docs

        self.v = TfidfVectorizer(input='content',
                            encoding='utf-8', decode_error='replace', strip_accents='unicode',
                            lowercase=True, analyzer='word', stop_words='english',
                            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                            ngram_range=(1, 2), max_features=self.num,
                            norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                            max_df=self.max_df, min_df=self.min_df)

        self.pdb = get_papers_db(flag='r')

    def make_corpus(self, training: bool):
        assert isinstance(training, bool)

        # determine which papers we will use to build tfidf
        if training and 0 < self.max_docs < len(self.pdb):
            # crop to a random subset of papers
            keys = list(self.pdb.keys())
            shuffle(keys)
            keys = keys[:self.max_docs]
        else:
            keys = self.pdb.keys()

        # yield the abstracts of the papers
        for p in keys:
            d = self.pdb[p]
            author_str = ' '.join([a['name'] for a in d['authors']])
            yield ' '.join([d['title'], d['summary'], author_str])

    def train(self):

        print("training tfidf vectors...")
        self.v.fit(self.make_corpus(training=True))

        print("running inference...")
        x = self.v.transform(self.make_corpus(training=False)).astype(np.float32)
        print(x.shape)

        print("saving to features to disk...")
        features = {
            'pids': list(self.pdb.keys()),
            'x': x,
            'vocab': self.v.vocabulary_,
            'idf': self.v._tfidf.idf_,
        }
        save_features(features)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arxiv Daemon')
    parser.add_argument('-n', '--num', type=int, default=100, help='up to how many papers to fetch')
    parser.add_argument('-s', '--start', type=int, default=0, help='start at what index')
    parser.add_argument('-b', '--break_after', type=int, default=3, help='how many 0 new papers in a row would cause '
                                                                         'us to stop early? or 0 to disable.')
    parser.add_argument('-n', '--num_f', type=int, default=20000, help='number of tfidf features')
    parser.add_argument('--min_df', type=int, default=5, help='min df')
    parser.add_argument('--max_df', type=float, default=0.1, help='max df')
    parser.add_argument('--max_docs', type=int, default=-1, help='maximum number of documents to use when '
                                                                 'training tfidf, or -1 to disable')

    args = parser.parse_args()
    print(args)

    arxiv_engine = Arxiv(num=args.num, start=args.start, break_after=args.break_after)

    arxiv_engine.format_api_query()

    arxiv_engine.fetch_papers()

    comp = Compute(num=args.num_f, min_df=args.min_df, max_df=args.max_df, max_docs=args.max_docs)

    comp.train()
