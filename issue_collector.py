import sys
import os
import datetime
from dateutil import parser as datetime_parser
import argparse
import logging
from github_wrapper import GithubAPIWrapper, MockGithub
from stores import MemoryStore, OverwriteStore, AppendStore

parser = argparse.ArgumentParser(description='Download data for processing')
parser.add_argument('-user', type=str, help='username for github API')
parser.add_argument('-password', type=str, help='password for github API')
parser.add_argument('--test', dest='test', action='store_true', help='only run tests')

logging.basicConfig(level=logging.INFO, filename="collection/issues_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class IssueCollector:
    LABELS = ["enhancement", "bug", "feature", "question"]
    GITHUB_LIMIT = 1000
    API_THRESHOLD = 10
    WAIT_SECS = 600

    def save_all(self, result, limit=None):
        logging.info("Saving all %d results" % result.totalCount)
        count = 0
        for elem in result:
            if count == limit:
                break
            self.store.store(elem.raw_data)
            count += 1
        return count

    def enqueue(self, obj):
        self.queue.append(obj)

    def enqueue_all(self, L):
        self.queue += L

    def dequeue(self):
        val = self.queue[0]
        self.queue = self.queue[1:]
        return val

    def queue_to_json(self):
        res = []
        for (label, low, high) in self.queue:
            res.append((label, str(low), str(high)))
        return res

    def queue_from_json(self, json_obj):
        self.queue = []
        for (label, low, high) in json_obj:
            self.queue.append((label, datetime_parser.parse(low).date(),
                datetime_parser.parse(high).date()))

    def save_queue(self):
        self.queue_store.store(self.queue_to_json())

    def load_queue(self):
        serialized_queue = self.queue_store.load()
        self.queue_from_json(serialized_queue)

    def execute_all(self):
        while len(self.queue) > 0:
            self.execute_once()

    def execute_once(self):
        self.find_all(self.dequeue())
        self.save_queue()

    def gen_upper_bisect(self, label, low, high):
        return (label, low + (high - low)/2 + datetime.timedelta(days=1), high)

    def gen_lower_bisect(self, label, low, high):
        return (label, low, low + (high - low)/2)

    def find_all(self, (label, low, high)):
        logging.info("Descending into: (%s, [%s, %s])" % (label, low, high))
        query_result = self.api.issues_by_date(label, low, high)
        total_count = query_result.totalCount
        if total_count <= self.GITHUB_LIMIT:
            self.save_all(query_result)
            return
        if total_count <= 2 * self.GITHUB_LIMIT:
            saved = self.save_all(query_result)
            query_result = self.api.issues_by_date(label, low, high)
            self.save_all(query_result, limit=saved)
            return
        if low == high:
            logging.warn("Can not bisect further: for label (%s,"
                    " [%s, %s]) there are %d issues"
                    % (label, low, high, total_count))
            return
        self.enqueue(self.gen_upper_bisect(label, low, high))
        self.enqueue(self.gen_lower_bisect(label, low, high))
        self.log_diag()

    def __init__(self, api, store, queue_store):
        self.api = api
        self.store = store
        self.queue_store = queue_store
        self.repos = []
        self.queue = []

    def log_diag(self):
        logging.info("API status: %s", self.api.get_api_status())
        logging.info("Current rate limit (QPH): %s", self.api.get_rate_limit())

def main(argv):
    args = parser.parse_args()

    cur_time = datetime.datetime.now().isoformat()
    filename = "issues." + cur_time
    queuename = "collection/issues_queue"
    relpath = "collection/" + filename
    latestrelpath = "collection/issues.latest"

    if os.path.exists(latestrelpath):
        os.unlink(latestrelpath)
    os.symlink(filename, latestrelpath)
    gh = GithubAPIWrapper(args.user, args.password)
    if os.path.exists(queuename):
        rc = IssueCollector(gh, AppendStore(relpath), OverwriteStore(queuename))
        rc.load_queue()
    else:
        rc = IssueCollector(gh, AppendStore(relpath), OverwriteStore(queuename))
        for label in IssueCollector.LABELS:
            rc.enqueue((label, datetime.date(2000, 1, 1), datetime.date(2014, 4, 7)))
    rc.execute_all()

if __name__ == "__main__":
    main(sys.argv)
