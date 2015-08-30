import requests
import re
from whoosh.index import create_in, open_dir
from whoosh.fields import * 
from whoosh.query import *
from whoosh.qparser import QueryParser
from whoosh import highlight
import lists
import textwrap

# Write index
def write(path):
	schema = Schema(quote=TEXT(stored=True), affiliation=TEXT(stored=True), force=TEXT(stored=True), name=TEXT(stored=True), count=NUMERIC(stored=True))
	ix = create_in(path + "\\Index", schema)
	writer = ix.writer()

	# Bring down HTML
	r1 = requests.get("http://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html")
	r2 = requests.get("http://www.imsdb.com/scripts/Star-Wars-The-Empire-Strikes-Back.html")
	r3 = requests.get("http://www.imsdb.com/scripts/Star-Wars-Return-of-the-Jedi.html")
	# r4 = requests.get("http://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html")
	r5 = requests.get("http://www.imsdb.com/scripts/Star-Wars-Attack-of-the-Clones.html")
	# r6 = requests.get("http://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html")
	data = r1.text + r2.text + r3.text# + r4.text + r5.text + r6.text

	quote_counter = 0
	sections = re.split('<b>', data)

	for section in sections:
		quote_counter += 1
		match = re.search('([A-Z])+(( )*([A-Z])*)*((\n)(.)+)+', section, re.MULTILINE)
		if match:
			string = match.group(0)
			string = re.sub('\n',' ', string)
			split = re.split('</b>', string)
			name = str(split[0].strip())
			quote = textwrap.dedent(str(split[1].strip()))
			quote = re.sub('  ','',quote)
			end_paren = ')'
			if end_paren in quote:
				quote = re.sub('\(.*\)','', quote)
			for char in lists.characters:
				for charName in char['Name']:
					if name.lower() == charName.lower():
						nameFirst = char['Name'][0].upper()
						affil = char['Affiliation']
						forceType = char['Force']
						writer.add_document(quote = quote, affiliation = affil, force = forceType, name = nameFirst, count = quote_counter)
	writer.commit()

# Loop to fill counter hashes
def countThings(ind, parser, listType, countType):
	for element in listType:
		if listType == lists.characters:
			countType[element["Name"][0]] = 0
			for name in element["Name"]:
				query = parser.parse(name)
				with ind.searcher() as searcher:
					results = searcher.search(query, limit=None)
					countType[element["Name"][0]] += len(results)
		else:
			query = parser.parse(element)
			with ind.searcher() as searcher:
				results = searcher.search(query, limit=None)
				countType[element] = len(results)


# Count of characters, places and vehicles
def counts(ind):
	countParser = QueryParser("quote", schema=ind.schema)
	charCounts, placeCounts, vehiCounts = {}, {}, {}

	countThings(ind, countParser, lists.characters, charCounts)
	countThings(ind, countParser, lists.places, placeCounts)
	countThings(ind, countParser, lists.vehicles, vehiCounts)

	print(charCounts)
	print(placeCounts)
	print(vehiCounts)

# Search index
def search(path):
	ix = open_dir(path + "\\Index")
	print(ix.doc_count())
	counts(ix)

if __name__ == "__main__":
	path = "\\starwars"
	# write(path)
	search(path)