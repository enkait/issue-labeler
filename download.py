import sys
import os
import datetime
import argparse
from github import Github
from github.GithubException import GithubException
import pprint
import json
import logging
import time
from collections import defaultdict

parser = argparse.ArgumentParser(description='Download data for processing')
parser.add_argument('-user', type=str, help='username for github API')
parser.add_argument('-password', type=str, help='password for github API')
parser.add_argument('--test', dest='test', action='store_true', help='only run tests')

logging.basicConfig(level=logging.INFO, filename="download_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class GithubAPIWrapper:
    def __init__(self, user, password):
        self.g = Github(user, password)

    def get_rate_limit(self):
        return self.g.get_rate_limit().rate.remaining

    def get_api_status(self):
        return self.g.get_api_status().status

    def query_by_stars(self, low, high):
        return self.g.search_repositories("stars:%d..%d" % (low, high))

class MockGithub:
    GITHUB_LIMIT = 1000

    def __init__(self):
        self.api_rate_limit = 30
        self.repositories = defaultdict(list)
        for i in range(50000):
            for j in range(20):
                self.repositories[i].append("%d/%d" % (i, j))

    def get_rate_limit(self):
        return self.api_rate_limit

    def get_api_status(self):
        return "OK"

    def get_repos(self, low, high):
        L = []
        for i in range(low, high+1):
            L += self.repositories[i]
        return L

    def query_by_stars(self, low, high):
        if self.api_rate_limit == 0:
            raise Exception("API rate limit hit")
        self.api_rate_limit -= 1

        class QueryResult:
            def __init__(self, L, limit):
                self.L = L
                self.totalCount = len(L)
                self.limit = limit

            def __iter__(self):
                if len(self.L) > self.limit:
                    raise Exception("Code shouldn't iterate when there are more than %d results" % (self.limit,))
                return iter(self.L)

        return QueryResult(self.get_repos(low, high), self.GITHUB_LIMIT)

    def sleep(self, secs):
        self.api_rate_limit = 30

class Sleeper:
    def sleep(self, secs):
        time.sleep(secs)

class FileStore:
    def __init__(self, file_path):
        self.f = open(file_path, "a+")

    def store(self, element):
        self.f.write(json.dumps(element.raw_data)+"\n")

class MemoryStore:
    def __init__(self):
        self.L = []

    def store(self, element):
        self.L.append(element)

    def get_stored(self):
        return self.L

class RepoCollector:
    labels = ["enhancement", "bug", "feature", "question"]
    LOW_STARS = 100
    HIGH_STARS = 100000
    GITHUB_LIMIT = 1000
    API_THRESHOLD = 10
    WAIT_SECS = 600

    def wait_api(self):
        while self.api.get_rate_limit() < self.API_THRESHOLD:
            logging.info("Waiting for %d seconds" % (self.WAIT_SECS))
            self.sleeper.sleep(self.WAIT_SECS)
        self.log_diag()

    def save_all(self, result):
        for elem in result:
            self.store.store(elem)

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
        self.find_all((high + low) / 2 + 1, high)
        self.find_all(low, (high + low) / 2)

    def find_good(self):
        self.find_all(self.LOW_STARS, self.HIGH_STARS)

    def __init__(self, api, store, sleeper = Sleeper()):
        self.api = api
        self.sleeper = sleeper
        self.store = store
        self.repos = []

        #print "Title:"
        #pprint.pprint(issue.title)
        #print "User:"
        #pprint.pprint(issue.user.login)
        #print "Labels:"
        #for label in issue.labels:
        #    pprint.pprint(label.name)
        #print "Body:"
        #pprint.pprint(issue.body)
        #TODO: escape before saving/replace with spaces

        print self.api.query_by_stars(100, 1000).totalCount

        #for repo in self.g.search_repos():
        #    try:
        #        output.write(json.dumps(repo.raw_data) + "\n")
        #    except GithubException as ex:
        #        logging.exception("Exception received")

        #for repo in self.g.get_repos():
        #    try:
        #        output.write(json.dumps(repo.raw_data) + "\n")
        #    except GithubException as ex:
        #        logging.exception("Exception received")

        #for label in self.labels:
        #    for issue in g.search_issues("label:" + label + " comments:>0"):
        #        try:
        #            output.write(json.dumps(issue.raw_data) + "\n")
        #        except GithubException as ex:
        #            logging.exception("Exception received")
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
        rc = RepoCollector(mg, ms, mg)
        rc.find_all(10, 40000)
        result = ms.get_stored()
        expected = mg.get_repos(10, 40000)
        if len(result) != len(expected):
            raise Exception("Length of result and expected differs")
        if set(result) != set(expected):
            raise Exception("Contents of result and expected differs")
        print "OK"
        return

    filename = "repos." + datetime.datetime.now().isoformat()
    relpath = "collection/" + filename
    latestrelpath = "collection/repos.latest"

    if os.path.exists(latestrelpath):
        os.unlink(latestrelpath)
    os.symlink(filename, latestrelpath)
    gh = GithubAPIWrapper(args.user, args.password)
    rc = RepoCollector(gh, FileStore(relpath))
    rc.find_good()

if __name__ == "__main__":
    main(sys.argv)
