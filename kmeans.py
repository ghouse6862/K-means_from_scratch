import pandas as pd
import numpy as np
import re
import random
import requests
from zipfile import ZipFile

class KMeans:
    def __init__(self,train,k):
        raw_input = pd.read_csv(train,sep='|',header=None,names=['Tweet_id','TimeStamp','Tweet'])
        self.df = self.pre_process(raw_input)
        self.k = k
        self.centers = random.sample(range(0,len(self.df)),self.k)
        self.clusters = [[]]*self.k
        
    def pre_process(self,raw_input):
        raw_input.drop(['Tweet_id','TimeStamp'],axis=1,inplace=True)
        raw_input['Tweet'] = raw_input['Tweet'].apply(self.process)
        return raw_input
        
    def process(self,tweet):
        regex1 = re.compile(r'\w*://\S*')
        if re.findall(r'\w*://\S*',tweet):
            for x in re.findall(r'\w*://\S*',tweet):
                tweet = regex1.sub(lambda m: m.group().replace(x,""),tweet).strip()
        regex2 = re.compile(r'@\S*')
        if re.findall(r'@\S*',tweet):
            for x in re.findall(r'@\S*',tweet):
                tweet = regex2.sub(lambda m: m.group().replace(x,""),tweet).strip()
        regex3 = re.compile(r'#')
        if re.findall(r'#',tweet):
            for x in re.findall(r'#',tweet):
                tweet = regex3.sub(lambda m: m.group().replace(x,""),tweet).strip()
        regex4 = re.compile(r"'")
        if re.findall(r"'",tweet):
            for x in re.findall(r"'",tweet):
                tweet = regex4.sub(lambda m: m.group().replace(x,""),tweet).strip()
        return tweet.lower()
    
    def jaccard_distance(self,st1,st2):
        a = st1.split()
        b = st2.split()
        match = 0
        for x in a:
            for y in b:
                if x == y:
                    match = match+1
        union = len(a) + len(b) - match
        j = 1 - (match/union)
        return j
    
    def SSE(self):
        sse = 0
        for cluster in self.clusters:
            for point in cluster:
                if point != cluster[0]:
                    sse = sse + (self.jaccard_distance(self.df['Tweet'][cluster[0]],self.df['Tweet'][point]) ** 2)
    
        return sse
    
    def centroid(self):
        new_centres = []
        for cluster in self.clusters:
            c = []
            for point in cluster:
                distance = 0
                for p in cluster:
                    distance = distance+self.jaccard_distance(self.df['Tweet'][point],self.df['Tweet'][p])
                c.append(distance)
            
            new_centres.append(cluster[c.index(min(c))])
        return new_centres
    
    def fit(self):
        change = True
        while change != False:
            for i,x in enumerate(self.centers):
                self.clusters[i] = [x]
            for i,x in enumerate(self.df['Tweet']):
                if i not in self.centers:
                    d = []
                    for y in self.centers:
                        d.append(self.jaccard_distance(self.df['Tweet'][y],x))
                    self.clusters[d.index(min(d))].append(i)
            new_centres = self.centroid()
            sse = self.SSE()
            print("SSE: {}".format(sse))
            print("Centers: {}".format(self.centers))
            if new_centres == self.centers:
                print("Done!")
                change = False
            else:
                self.centers = new_centres
                self.clusters = [[]]*self.k
                
    def print_centers(self):
        i = 0
        for x in self.centers:
            print("center {}:".format(i))
            print(self.df['Tweet'][x])
            i += 1
            
    def cluster_size(self):
        i = 0
        for cluster in self.clusters:
            print("cluster {}: {} Tweets".format(i,len(cluster)))
            i+=1
            
            
if __name__ == "__main__":
    print("Enter the number of clusters to make: ")
    num = int(input())
    
    r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00438/Health-News-Tweets.zip")
    file_name = "Health-News-Tweets.zip"
    
    with open("Health-News-Tweets.zip",'wb') as f:
        f.write(r.content)
        
    with ZipFile(file_name, 'r') as zip:
        zip.extractall()
  
    km = KMeans('Health-Tweets/usnewshealth.txt',num)
    km.fit()
    km.print_centers()
    print("")
    km.cluster_size()