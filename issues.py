import sys
import os
import datetime
import argparse
import logging
import json
from github_wrapper import GithubAPIWrapper, MockGithub
from stores import MemoryStore, OverwriteStore, AppendStore

parser = argparse.ArgumentParser(description='Download data for processing')
parser.add_argument('-user', type=str, help='username for github API')
parser.add_argument('-password', type=str, help='password for github API')
parser.add_argument('--test', dest='test', action='store_true', help='only run tests')

logging.basicConfig(level=logging.INFO, filename="download_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class IssueCollector:
    LABELS = ["enhancement", "bug", "feature", "question"]
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
        repo = self.dequeue()
        self.find_all(repo)
        self.save_queue()

    def find_all(self, repo):
        logging.info("Getting issues for: %s" % (repo.full_name))
        for label in repo.get_labels():
            if label.name in self.LABELS:
                self.wait_api()
                self.save_all(repo.get_issues(state="all", labels=[label]))

    def __init__(self, api, store, queue_store):
        self.api = api
        self.store = store
        self.queue_store = queue_store
        self.queue = []
        self.log_diag()

    def log_diag(self):
        logging.info("API status: %s", self.api.get_api_status())
        logging.info("Current rate limit (QPH): %s", self.api.get_rate_limit())

def get_repos(repo_file):
    with open(repo_file) as f:
        yield json.loads(f.readline().strip())

def main(argv):
    args = parser.parse_args()

    cur_time = datetime.datetime.now().isoformat()
    filename = "issues." + cur_time
    repo_filename = "repolist"
    queuename = "issue_queue"
    relpath = "collection/" + filename
    latestrelpath = "collection/issues.latest"

    if os.path.exists(latestrelpath):
        os.unlink(latestrelpath)
    os.symlink(filename, latestrelpath)
    gh = GithubAPIWrapper(args.user, args.password)
    if os.path.exists(queuename):
        ic = IssueCollector(gh, AppendStore(relpath), OverwriteStore(queuename))
        ic.load_queue()
    else:
        ic = IssueCollector(gh, AppendStore(relpath), OverwriteStore(queuename))
        ic.enqueue_all(map(gh.load_repo, get_repos(repo_filename)))
    print "OMG"
    ic.execute_all()

if __name__ == "__main__":
    main(sys.argv)