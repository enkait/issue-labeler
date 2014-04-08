import json
import time
from collections import defaultdict
from github import Github
from github.GithubException import GithubException
from github.Repository import Repository

class GithubAPIWrapper:
    def __init__(self, user, password):
        self.g = Github(user, password)

    def get_rate_limit(self):
        return self.g.get_rate_limit().rate.remaining

    def get_api_status(self):
        return self.g.get_api_status().status

    def query_by_stars(self, low, high):
        return self.g.search_repositories("stars:%d..%d" % (low, high))

    def issues_by_date(self, label, low, high):
        return self.g.search_issues("created:%s..%s type:issue label:%s" % (low, high, label))

    def load_repo(self, raw):
        return self.g.create_from_raw_data(Repository, raw)

    def sleep(self, secs):
        time.sleep(secs)

class MockGithub:
    GITHUB_LIMIT = 1000

    def __init__(self):
        self.api_rate_limit = 300
        self.repositories = defaultdict(list)
        class MockRepo:
            def __init__(self, parent, raw_data):
                self.parent = parent
                self._raw_data = raw_data

            @property
            def raw_data(self):
                if self.parent.api_rate_limit <= 0:
                    raise Exception("Code shouldn't iterate when there are no more api calls")
                self.parent.api_rate_limit -= 1
                return self._raw_data

        for i in range(10100):
            for j in range(7):
                self.repositories[i].append(MockRepo(self, "%d/%d" % (i, j)))
        print "Done"

    def get_rate_limit(self):
        return self.api_rate_limit

    def get_api_status(self):
        return "OK"

    def get_repos(self, low, high):
        L = []
        for i in range(low, high+1):
            L += self.repositories[i]
        return L

    def get_raw_repos(self, low, high):
        L = self.get_repos(low, high)
        return [o._raw_data for o in L]

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
        self.api_rate_limit = 300
