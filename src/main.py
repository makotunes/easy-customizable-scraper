#!/root/anaconda3/bin python

from scraper import ScannerWrapper
import argparse
import sys

def get_args():

    parser = argparse.ArgumentParser()

    if sys.stdin.isatty():
        parser.add_argument(
            "entry_url", help="(Required) Initial url to start to scan. sub paths of specified one are scanned recursively.", type=str)

        parser.add_argument("--document-xpath",
                            help="Xpath for root node for target range.", type=str)
        parser.add_argument(
            "--image-xpath", help="Xpath to get a image path.", type=str)
        parser.add_argument(
            "--allow-rule", help="Allow filter rule of target urls.", type=str)
        parser.add_argument(
            "--deny-rule", help="Deny filter rule of target urls.", type=str)
        parser.add_argument(
            "--page-limit", help="Scaned limittation of number of pages. -1 means unlimited number.", type=int)
        parser.add_argument(
            "--exclude-reg", help="Excluded words with regular expression for morphological analysis.", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    entry_url = args.entry_url
    document_xpath = args.document_xpath
    image_xpath = args.image_xpath
    allow_rule = args.allow_rule
    deny_rule = args.deny_rule
    page_limit = args.page_limit
    exclude_reg = args.exclude_reg

    #exclude_reg = re.compile(
    #    "\d(年|月|日|時|分|秒|ｇ|\u4eba|\u672c|cm|ml|g|\u5206\u679a\u5ea6)|hacarusinc|allrightsreserved")

    scanner_wrapper = ScannerWrapper(
        entry_url, document_xpath, image_xpath, allow_rule, deny_rule, page_limit, exclude_reg)
    scanner_wrapper.main()
