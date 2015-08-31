import requests, re, string
from whoosh.index import create_in, open_dir
from whoosh.fields import * 
from whoosh.query import *
from whoosh.qparser import QueryParser
from whoosh import highlight
import lists
import textwrap, os, math, numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import lda
from stop_words import get_stop_words

# Write index
def write(path):
	schema = Schema(quote=TEXT(stored=True), affiliation=TEXT(stored=True), force=TEXT(stored=True), name=TEXT(stored=True), count=NUMERIC(stored=True))
	ix = create_in(path + "\\Index", schema)
	writer = ix.writer()

	# Bring down HTML
	r = requests.get("http://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html")
	data = r.text

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

# All quotes to text file, depending on affiliation
def affiliations(ind):
	affils = [u"Rebel Alliance", u"Empire", u"None"]
	affilParser = QueryParser("affiliation", schema=ind.schema)

	for affil in affils:
		query = affilParser.parse(affil)

		with ind.searcher() as searcher:
			results = searcher.search(query, limit=None)
			with open('SW ' + str(affil) + '.txt','w') as f:
				f.write("Result count: " + str(len(results)) + "\n")
				for result in results:
					f.write(str(result['quote']) + "\n")

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

def charQuotes(ind):
	charParser = QueryParser('name', schema=ind.schema)
	for character in lists.characters:
		query = charParser.parse(character["Name"][0])
		with ind.searcher() as searcher:
			results = searcher.search(query, limit=None)
			with open('Characters\\' + str(character["Name"][0]) + '.txt','w') as f:
				for result in results:
					f.write(str(result['quote']) + "\n")

def outputSentiment(ind, path):
	charFiles = os.listdir("Characters")
	stanfordPath = 'Text Analytics\\stanford-corenlp-full-2014-08-27'
	for charFile in charFiles:
		os.system("java -cp \"" + stanfordPath + "\\*\" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file \"" + path + "\\Characters\\" + charFile +
			"\" > \"" + path + "\\CharSent\\" + charFile +"\"")

def averageSentiment(path):
	with open(path, 'r') as f:
		raw = f.read()
		neg_count = Counter(re.findall(r"Negative", raw))
		very_neg_count = Counter(re.findall(r"Very negative", raw))
		pos_count = Counter(re.findall(r"Positive", raw))
		very_pos_count = Counter(re.findall(r"Very positive", raw))
		neut_count = Counter(re.findall(r"Neutral", raw))
	
		total = pos_count["Positive"] + neg_count["Negative"] + neut_count["Neutral"] + very_pos_count["Very positive"] + very_neg_count["Very negative"]
		sumSent = (pos_count["Positive"] - neg_count["Negative"]) + 3*(very_pos_count["Very positive"] - very_neg_count["Very negative"])
		if total == 0:
			return 0
		else:
			return (sumSent / total)


def charSentiment(ind, path):
	# charQuotes(ind)
	# outputSentiment(ind,path)
	sentFiles = os.listdir(path + "\\CharSent")
	chars = {}
	for sentFile in sentFiles:
		avgSent = averageSentiment(path + "\\CharSent\\" + sentFile)
		chars[sentFile[:-4]] = avgSent
	return chars

# Retrieve variables on which to cluster
def clusterVars(ind, path):
	chars = []

	# Importance (# lines of dialogue)
	nameParser = QueryParser("name", schema=ind.schema)
	for character in lists.characters:
		query = nameParser.parse(character["Name"][0])
		with ind.searcher() as searcher:
			results = searcher.search(query, limit=None)
			obj = {}
			obj["Name"] = character["Name"][0]
			obj["NumQuotes"] = 0
			for result in results:
				obj["NumQuotes"] += len(result['quote'].split('\n'))
			chars.append(obj)
	
	# Interest in Luke (Luke mentions), Force / Dark Side mentions
	query1 = Or([Phrase("quote", ["red","five"]),Term("quote","luke"),Term("quote","skywalker")])
	query2 = Or([Phrase("quote",["dark","side"]),Term("quote","force")])

	with ind.searcher() as searcher:
		results1 = searcher.search(query1, limit=None)
		results2 = searcher.search(query2, limit=None, terms=True)
		for character in chars:
			character["InterestInLuke"] = 0
			character["ForceMentions"] = 0
			for result1 in results1:
				if result1['name'] == character["Name"].upper():
					character["InterestInLuke"] += 1
			for result2 in results2:
				if result2['name'] == character["Name"].upper():
					character["ForceMentions"] += 1

	# Take proportions
	for character in chars:
		if character["NumQuotes"] > 0:
			character["InterestInLuke"] = (character["InterestInLuke"] + 0.0) / character["NumQuotes"]
			character["ForceMentions"] = (character["ForceMentions"] + 0.0) / character["NumQuotes"]

	# Character sentiment average
	charSents = charSentiment(ind, path)
	for charSent in charSents:
		for character in chars:
			if character["Name"] == charSent:
				character["Sentiment"] = charSents[charSent]

	return chars

# Transform cluster variables into an array and normalise
def toArray(data):
	arr = []

	# Find mins and max's for min-max normalisation
	minArr = [data[0]["NumQuotes"],data[0]["InterestInLuke"],data[0]["ForceMentions"],data[0]["Sentiment"]]
	maxArr = [data[0]["NumQuotes"],data[0]["InterestInLuke"],data[0]["ForceMentions"],data[0]["Sentiment"]]

	for character in data:
		if character["NumQuotes"] < minArr[0]:
			minArr[0] = character["NumQuotes"]
		if character["NumQuotes"] > maxArr[0]:
			maxArr[0] = character["NumQuotes"]
		if character["InterestInLuke"] < minArr[1]:
			minArr[1] = character["InterestInLuke"]
		if character["InterestInLuke"] > maxArr[1]:
			maxArr[1] = character["InterestInLuke"] 
		if character["ForceMentions"] < minArr[2]:
			minArr[2] = character["ForceMentions"]
		if character["ForceMentions"] > maxArr[2]:
			maxArr[2] = character["ForceMentions"]
		if character["Sentiment"] < minArr[3]:
			minArr[3] = character["Sentiment"]
		if character["Sentiment"] > maxArr[3]:
			maxArr[3] = character["Sentiment"]

	for character in data:
		numQuotes = (character["NumQuotes"] - minArr[0]) / (maxArr[0]-minArr[0])
		lukeInterest = (character["InterestInLuke"] - minArr[1]) / (maxArr[1]-minArr[1])
		forceMentions = (character["ForceMentions"] - minArr[2]) / (maxArr[2]-minArr[2])
		sentiment = (character["Sentiment"] - minArr[3]) / (maxArr[3]-minArr[3])
		arr.append([numQuotes,lukeInterest,forceMentions,sentiment])

	return arr


def kmeans(data, clusters=5):
	xData = toArray(data)
	clust = KMeans(n_clusters=clusters)
	clust.fit(xData)

	# print(data)
	print(clust.cluster_centers_)

	# Output in a table, with heading cluster number and names below
	allocate = {}
	for label in clust.labels_:
		allocate[str(label)] = ""

	for i, label in enumerate(clust.labels_):
			allocate[str(label)] += data[i]["Name"] + ", "

	for c in allocate:
		print(c + ": " + allocate[c])


# Create term-document matrix
def termDoc(ind):

	# Crawl index to find all terms
	terms = set()
	nameParser = QueryParser("name", schema=ind.schema)
	stopwords = get_stop_words('en')
	for character in lists.characters:
		query = nameParser.parse(character["Name"][0])
		with ind.searcher() as searcher:
			results = searcher.search(query, limit=None)
			for result in results:
				tokens = result['quote'].split(' ')
				for token in tokens:
					term = token.strip(string.punctuation).lower()
					if term not in stopwords:
						terms.add(term)

	# Crawl again to assign to term-document matrix
	docs = ind.doc_count()
	TD = np.zeros((docs,len(terms)), dtype=int)
	termsArr = list(terms)

	for character in lists.characters:
		query = nameParser.parse(character["Name"][0])
		with ind.searcher() as searcher:
			results = searcher.search(query, limit=None)
			for doc, result in enumerate(results):
				tokens = result['quote'].split(' ')
				for token in tokens:
					term = token.strip(string.punctuation).lower()
					try:
						termIndex = termsArr.index(term)
					except:
						continue
					TD[doc, termIndex] += 1

	return (TD, termsArr)

# LDA
def topics(ind, num_topics=20, num_top_words=10):
	TD, terms = termDoc(ind)
	model = lda.LDA(n_topics=numtopics, n_iter=1000, random_state=1)
	model.fit(TD)
	topic_word = model.topic_word_
	n_top_words = num_top_words
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(terms)[np.argsort(topic_dist)][:-n_top_words:-1]
		print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# Run some analysis
def run(path):
	ix = open_dir(path + "\\Index")
	# print(ix.doc_count())

	affiliations(ix)
	counts(ix)
	clusterVariables = clusterVars(ix, path)
	kmeans(clusterVariables)

	topics(ix)

if __name__ == "__main__":
	path = "\\7 starwars"
	write(path)
	run(path)