import sys
import os
import datetime
import argparse
from github import Github
from github.GithubException import GithubException
import pprint
import json
import logging

parser = argparse.ArgumentParser(description='Download data for processing')
parser.add_argument('-user', type=str, help='username for github API')
parser.add_argument('-password', type=str, help='password for github API')

logging.basicConfig(level=logging.INFO, filename="download_log", filemode="a+",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class IssueCollector:
    labels = ["enhancement", "bug", "feature", "question"]

    def __init__(self, output, user, password):
        g = self.g = Github(user, password)
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

        print self.g.search_repositories("stars:>100").totalCount

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
        print "API status:", self.g.get_api_status().status
        print "Current rate limit (QPH):", self.g.get_rate_limit().rate.limit

def main(argv):
    args = parser.parse_args()

    filename = "output." + datetime.datetime.now().isoformat()
    relpath = "collection/" + filename
    latestrelpath = "collection/output.latest"

    with open(relpath, "w") as output:
        if os.path.exists(latestrelpath):
            os.unlink(latestrelpath)
        os.symlink(filename, latestrelpath)
        ic = IssueCollector(output, args.user, args.password)

if __name__ == "__main__":
    main(sys.argv)
