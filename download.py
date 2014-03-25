import sys
import os
import datetime
import argparse
import logging
from github_wrapper import GithubAPIWrapper, MockGithub
from stores import MemoryStore, OverwriteStore, AppendStore

parser = argparse.ArgumentParser(description='Download data for processing')
parser.add_argument('-user', type=str, help='username for github API')
parser.add_argument('-password', type=str, help='password for github API')
parser.add_argument('--test', dest='test', action='store_true', help='only run tests')

logging.basicConfig(level=logging.INFO, filename="download_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class RepoCollector:
    labels = ["enhancement", "bug", "feature", "question"]
    GITHUB_LIMIT = 1000
    API_THRESHOLD = 10
    WAIT_SECS = 600

    def wait_api(self):
        # TODO: watch out for separate rate limit for search and other
        while self.api.get_rate_limit() < self.API_THRESHOLD:
            logging.info("Waiting for %d seconds" % (self.WAIT_SECS))
            self.api.sleep(self.WAIT_SECS)
        self.log_diag()

    def save_all(self, result):
        for elem in result:
            self.wait_api()
            self.store.store(elem.raw_data)

    def enqueue(self, low, high):
        self.queue.append((low, high))

    def enqueue_all(self, L):
        self.queue += L

    def dequeue(self):
        val = self.queue[0]
        self.queue = self.queue[1:]
        return val

    def save_queue(self):
        self.queue_store.store(self.queue)

    def load_queue(self):
        self.queue = self.queue_store.load()

    def execute_all(self):
        while len(self.queue) > 0:
            self.execute_once()

    def execute_once(self):
        low, high = self.dequeue()
        self.find_all(low, high)
        self.save_queue()

    def find_all(self, low, high):
        logging.info("Descending into: [%d, %d]" % (low, high))
        self.wait_api()
        query_result = self.api.query_by_stars(low, high)
        if query_result.totalCount <= self.GITHUB_LIMIT:
            self.save_all(query_result)
            return
        if low == high:
            logging.warn("Can not dissect further: %d stars has %d results" % (low, query_result.totalCount))
            return
        self.enqueue((high + low) / 2 + 1, high)
        self.enqueue(low, (high + low) / 2)

    def __init__(self, api, store, queue_store):
        self.api = api
        self.store = store
        self.queue_store = queue_store
        self.repos = []
        self.queue = []
        self.log_diag()

    def log_diag(self):
        logging.info("API status: %s", self.api.get_api_status())
        logging.info("Current rate limit (QPH): %s", self.api.get_rate_limit())

def main(argv):
    args = parser.parse_args()

    if args.test:
        print "Testing"
        mg = MockGithub()
        ms = MemoryStore()
        qs = MemoryStore()
        rc = RepoCollector(mg, ms, qs)
        rc.enqueue(10, 10000)
        rc.execute_all()
        result = ms.get_stored()
        expected = mg.get_raw_repos(10, 10000)
        if len(result) != len(expected):
            raise Exception("Length of result and expected differs")
        if set(result) != set(expected):
            raise Exception("Contents of result and expected differs")
        print "OK"
        return

    cur_time = datetime.datetime.now().isoformat()
    filename = "repos." + cur_time
    queuename = "queue"
    relpath = "collection/" + filename
    latestrelpath = "collection/repos.latest"

    if os.path.exists(latestrelpath):
        os.unlink(latestrelpath)
    os.symlink(filename, latestrelpath)
    gh = GithubAPIWrapper(args.user, args.password)
    if os.path.exists(queuename):
        rc = RepoCollector(gh, AppendStore(relpath), OverwriteStore(queuename))
        rc.load_queue()
    else:
        rc = RepoCollector(gh, AppendStore(relpath), OverwriteStore(queuename))
        rc.enqueue(100, 100000)
    rc.execute_all()

if __name__ == "__main__":
    main(sys.argv)
