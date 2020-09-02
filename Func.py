import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import NearestNeighbors
import ast
import Levenshtein
import scipy
from joblib import dump, load
import random
import urllib.request
import json
import matplotlib.pyplot as plt
import warnings

with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)



def return_detail(imdbId):
    final = pd.read_csv('materials/final.csv')
    df = pd.read_csv('materials/Movie_rate.csv')
    #poster = pd.read_csv('materials/fastlink_poster.csv')
    name = pd.read_csv('materials/movie_Name.csv')
    res= {}
    res['poster'] = GetPoster_single(imdbId)
    res['imdbRating'],res['Genre'],res['Director'],res['Actors']  = final[final['imdbID'] == \
                                        imdbId][['imdbRating','Genre','Director','Actors']].values[0]
    res['Title'] = name[name['imdbID']==imdbId]['Title'].values[0]

    for i in ['Genre','Director']:
        res[i] = ', '.join(ast.literal_eval(res[i]))
    res['Actors'] = ast.literal_eval(res['Actors'])

    rate_tmp = df[df.imdb==imdbId][['Aged 18-29', 'Aged 30-44', 'Aged 45+','Aged under 18',
        'Females', 'Females Aged 18-29', 'Females Aged 30-44','Females Aged 45+', 'Females Aged under 18',
        'Males', 'Males Aged 18-29', 'Males Aged 30-44', 'Males Aged 45+',
       'Males Aged under 18','Non-US users','Top 1000 voters']].to_dict()

    def process_str(v,name):
        if type(list(v.values())[0]) == float:
            return 0
        else:
            return json.loads(list(v.values())[0].replace('\'','"'))[name]
    res['Top1000'] = process_str(rate_tmp['Top 1000 voters'],'aggregateRating')
    res['NonUS'] = process_str(rate_tmp['Non-US users'],'aggregateRating')

    ages = ['Aged under 18','Aged 18-29', 'Aged 30-44', 'Aged 45+','All Ages']

    data = {'ages'     : ages,
            'All': [],
            'Males'   : [],
            'Females' : []}

    for gender in ['Females','Males','All']:
        for age in ages:
            tmp_dic = {'Females':'Females ','Males':'Males ','All':''}
            if age == 'All Ages' and gender=='All':
                data[gender].append(res['imdbRating'])
            elif age == 'All Ages':
                data[gender].append(process_str(rate_tmp[gender],'aggregateRating'))
            else:
                data[gender].append(process_str(rate_tmp[tmp_dic[gender]+age],'aggregateRating'))

    x = np.array(list(range(len(data['ages']))))

    plt.switch_backend('Agg')

    plt.rcParams.update({'font.size': 30})
    fig,ax = plt.subplots(figsize=(23,10))
    ax.bar(x-0.2, data['Males'], width=0.2, color='b', align='center')
    ax.bar(x, data['Females'], width=0.2, color='g', align='center')
    ax.bar(x+0.2, data['All'], width=0.2, color='r', align='center')
    ax.set_xticklabels(['']+data['ages'])
    ax.set_ylim([0, 12])
    ax.set_title('Rate by Age')
    fig.legend([ 'Males', 'Females','All'],loc='upper right', bbox_to_anchor=(0.90, 0.85),ncol=3)
    fig.savefig('static/images/plot.png')
    plt.close(fig)
    return res

def ramdom_movies():
    movie_list=  np.load('materials/Total_movie_poster_list.npy',allow_pickle=True)
    four = random.sample(list(movie_list),4)
    return four

def GetPoster_single(tt):
    poster_list= pd.read_csv('materials/fastlink_poster.csv')
    return poster_list[poster_list['imdbID']==tt]['poster_path'].values[0]

def GetPoster(ll):
    poster_list= pd.read_csv('materials/fastlink_poster.csv')
    return poster_list[poster_list['imdbID'].isin(ll)]['poster_path'].to_list()

def cloest_name(s):
    name_list = np.load('materials/Total_movie_name_list.npy',allow_pickle=True)
    func = lambda t: Levenshtein.distance(s.lower(),t.lower())
    vfunc = np.vectorize(func)
    score = vfunc(name_list)
    return name_list[np.argmin(score)]

def get_movie_id(name):
    df = pd.read_csv('materials/movie_Name.csv')
    try:
        return df[df.Title == name.strip()]['imdbID'].values[0]
    except:
        return df[df.Title.apply(lambda x: x.lower()) == cloest_name(name)]['imdbID'].values[0]

def reco_knn(movie_list):
        df_final = pd.read_csv('materials/final.csv')
        df_name =  pd.read_csv('materials/movie_Name.csv')
        with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=UserWarning)
              nn = load('materials/knn.joblib')
        train = scipy.sparse.load_npz('materials/train.npz')

        m_locs = df_final[df_final.imdbID.isin(movie_list)].index.tolist()
        scaled_ratings = pd.DataFrame([5,5,5,5,5]) * 0.2
        weighted_avg_movie = scaled_ratings.values.reshape(1,-1).dot(train[m_locs,:].toarray()) / len(scaled_ratings)
        dists, indices = nn.kneighbors(weighted_avg_movie)
        tmp  = pd.merge(df_final.iloc[indices[0]][['imdbID']]\
                ,df_name,left_on='imdbID', right_on='imdbID', how='inner')

        return tmp[~tmp['imdbID'].isin(movie_list)]['imdbID'].to_list()[:20]

def reco_cb(movie_list):
        with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=UserWarning)
              cb = load('materials/cb.joblib')
        with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=UserWarning)
              dicv = load('materials/cb_dict.joblib')
        by_user_ratings = pd.read_csv('materials/groupbyed_ratings.csv')

        target_tmp = pd.Series([{i:1 for i in movie_list}])
        target = dicv.transform(target_tmp)
        dists, indices = cb.kneighbors(target)
        neighbors = [by_user_ratings.index[i] for i in indices[0]][1:]
        ratings_grp = by_user_ratings[by_user_ratings['userId'].isin(neighbors)][['imdbId','rating']]
        def bayes_sum(N, mu):
            return lambda x: (x.sum() + mu*N) / (x.count() + N)
        return ratings_grp.applymap(ast.literal_eval).apply(pd.Series.explode).groupby('imdbId')\
        ['rating'].aggregate(bayes_sum(5, 3)).sort_values(ascending=False).head(20)
def reco(movie_list):
    return reco_cb(movie_list).index[:3].tolist()+reco_knn(movie_list)[:2]
