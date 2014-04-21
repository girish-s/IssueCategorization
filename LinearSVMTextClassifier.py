from nltk.corpus import names
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import cross_validation

stopwordsString = "a, about, above, across, after, again, against, all, almost, alone, along, already, also, although, always, am, among, an, and, another, any, anybody, anyone, anything, anywhere, are, area, areas, aren't, around, as, ask, asked, asking, asks, at, away, b, back, backed, backing, backs, be, became, because, become, becomes, been, before, began, behind, being, beings, below, best, better, between, big, both, but, by, c, came, can, cannot, can't, case, cases, certain, certainly, clear, clearly, come, could, couldn't, d, did, didn't, differ, different, differently, do, does, doesn't, doing, done, don't, down, downed, downing, downs, during, e, each, early, either, end, ended, ending, ends, enough, even, evenly, ever, every, everybody, everyone, everything, everywhere, f, face, faces, fact, facts, far, felt, few, find, finds, first, for, four, from, full, fully, further, furthered, furthering, furthers, g, gave, general, generally, get, gets, give, given, gives, go, going, good, goods, got, great, greater, greatest, group, grouped, grouping, groups, h, had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll, her, here, here's, hers, herself, he's, high, higher, highest, him, himself, his, how, however, how's, i, i'd, if, i'll, i'm, important, in, interest, interested, interesting, interests, into, is, isn't, it, its, it's, itself, i've, j, just, k, keep, keeps, kind, knew, know, known, knows, l, large, largely, last, later, latest, least, less, let, lets, let's, like, likely, long, longer, longest, m, made, make, making, man, many, may, me, member, members, men, might, more, most, mostly, mr, mrs, much, must, mustn't, my, myself, n, necessary, need, needed, needing, needs, never, new, newer, newest, next, no, nobody, non, noone, nor, not, nothing, now, nowhere, number, numbers, o, of, off, often, old, older, oldest, on, once, one, only, open, opened, opening, opens, or, order, ordered, ordering, orders, other, others, ought, our, ours, ourselves, out, over, own, p, part, parted, parting, parts, per, perhaps, place, places, point, pointed, pointing, points, possible, present, presented, presenting, presents, problem, problems, put, puts, q, quite, r, rather, really, right, room, rooms, s, said, same, saw, say, says, second, seconds, see, seem, seemed, seeming, seems, sees, several, shall, shan't, she, she'd, she'll, she's, should, shouldn't, show, showed, showing, shows, side, sides, since, small, smaller, smallest, so, some, somebody, someone, something, somewhere, state, states, still, such, sure, t, take, taken, than, that, that's, the, their, theirs, them, themselves, then, there, therefore, there's, these, they, they'd, they'll, they're, they've, thing, things, think, thinks, this, those, though, thought, thoughts, three, through, thus, to, today, together, too, took, toward, turn, turned, turning, turns, two, u, under, until, up, upon, us, use, used, uses, v, very, w, want, wanted, wanting, wants, was, wasn't, way, ways, we, we'd, well, we'll, wells, went, were, we're, weren't, we've, what, what's, when, when's, where, where's, whether, which, while, who, whole, whom, who's, whose, why, why's, will, with, within, without, won't, work, worked, working, works, would, wouldn't, x, y, year, years, yes, yet, you, you'd, you'll, young, younger, youngest, your, you're, yours, yourself, yourselves, you've, z"

nos = [u'00', u'01', u'02', u'025', u'0284430014', u'0295793338', u'0295859654', u'0297881489', u'03', u'0387741595', u'0401953488', u'0402', u'0402208721', u'0402920336', u'0403354730', u'0404', u'0404723868', u'0410', u'0411', u'0411561813', u'0412', u'0412637238', u'0413199965', u'0413507925', u'0413759044', u'0415', u'0418', u'0422752994', u'0423', u'0423415575', u'0423476329', u'0431384839', u'0431605774', u'0434626518', u'0435912025', u'0435920012', u'0438', u'0439447099', u'0478089708', u'05021965', u'084868', u'10', u'110', u'115', u'12', u'1300300693', u'1300328587', u'1300555241', u'14', u'15', u'160', u'169', u'17', u'178', u'18', u'19', u'1934', u'1943', u'1951', u'1955', u'1956', u'1957', u'1961', u'1963', u'1965', u'1968', u'1973', u'1974', u'1976', u'1979', u'1985', u'1986', u'1988', u'1989', u'1991', u'1992', u'200', u'208', u'2160', u'22', u'24', u'240', u'25', u'255', u'27', u'288', u'30', u'31', u'32', u'33786', u'40', u'409', u'4169', u'450', u'48', u'50', u'52', u'57', u'59', u'60', u'627', u'63', u'64', u'679', u'69', u'72', u'721', u'723', u'73', u'782', u'799', u'82', u'839', u'85', u'85230429000206', u'868', u'87', u'88027747', u'89', u'90781974', u'915', u'94', u'970', u'97033786', u'98960620', u'992', u'996', u'9999']


def readData(dataFile):
    f = open(dataFile)

    lines = f.readlines()
    lines = [x.strip() for x in lines]
    lines = [x.lower() for x in lines]
    lines = [x.split(',') for x in lines]
    
    
    tmp = True
    Chats = []
    labels = []
    for line in lines:
        
        if len(line) < 4 or tmp or line[3] == '':
            tmp = False            
            continue
        tmp = False
        
        Chats.append(line[2])
        labels.append(line[3])

    return [Chats , labels]



def getTrainTestData(chatData , frac , minPts):
    x1 = getLabelCnts(chatData)
    x2 = [x[0] for x in x1 if x[1] >= minPts]

    tmp1 = [chatData[0][i] for i in range(len(chatData[0])) if chatData[1][i] in x2]
    tmp2 = [chatData[1][i] for i in range(len(chatData[1])) if chatData[1][i] in x2]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(tmp1 , tmp2 , train_size=frac , random_state=42)
    

    return dict([['trainchat',X_train] , ['trainlabels' , y_train] , ['testchat' , X_test] , ['testlabels' ,y_test]]) 


def getLabelCnts(t):
    return [[y,t[1].count(y)] for y in set(t[1])]
     


def learnSVMClassier(trainData , trainLabels):
    clf = LinearSVC()
    clf.fit(trainData , trainLabels)
    return clf


def createTfidfVectorizer(textcorpus):
    s1 = stopwordsString.split(', ')
    s1 = [unicode(x) for x in s1]
    s3 = [x.lower() for x in names.words()]
    s2 = stopwords.words()
    sw = s1+s2+s3+nos
    vectorizer = TfidfVectorizer(min_df=1 , stop_words=sw)
    vectorizer.fit_transform(textcorpus)
    return vectorizer



def testClassifier(Xtest , Ytest , clf):
    return clf.score(Xtest,Ytest)


def Predict(Xtest , clf):
    return clf.predict(Xtest)

if __name__ == '__main__':
    data = readData('C:\Users\girish.s\Desktop\DSGProjects\ChatTrends\Book3.csv')
    
    clldic = dict([[50,5],[40,10] , [30,15] ,[20,25]])
    npts = [50,40,30,20]

    for n in npts :
    
        splitData = getTrainTestData(data , 0.7 , n)

        vec = createTfidfVectorizer(splitData['trainchat'])

        Xtr = vec.transform(splitData['trainchat'])

        Ytr = splitData['trainlabels']

        clf = learnSVMClassier(Xtr , Ytr)

        Xtest = vec.transform(splitData['testchat'])
        Ytest = splitData['testlabels']

        accuracy = testClassifier(Xtest , Ytest , clf)
        print 'no of training samples : '+str(len(Ytr))
        print 'no of test samples : ' + str(len(Ytest))
        print 'accuracy with '+str(clldic[n])+' classes : '+str(accuracy)

        print '\n\n'
    


    












