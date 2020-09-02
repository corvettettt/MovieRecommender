from python.Func import *
from flask import Flask,render_template,request,redirect,url_for,session
import json
app = Flask(__name__)

data = DataStore()

app.secret_key = b'Knfiesz!1'
app.config['SECRET_KEY'] = 'Knfiesz!1'
@app.route('/Home',methods=['GET'])
def home():
    if request.method == 'GET':
        four_movies = ramdom_movies()
        styles = ["background-image:url("+i+");min-height: 400px; background-position: 50% 50%" for i in four_movies]
#        print(styles)
        return render_template('Home.html', movie1 =styles[0],movie2 =styles[1],\
                            movie3 =styles[2],movie4 =styles[3])

@app.route('/next_recoengine')
def next_recoengine():
    return redirect('/recoengine')



@app.route('/recoengine',methods=['GET','POST'])
def recoengine():
    if request.method == 'GET':
        return render_template('Recomendation.html')
    else:
        liked = []
        liked.append(get_movie_id(request.form['Movie_1']))
        liked.append(get_movie_id(request.form['Movie_2']))
        liked.append(get_movie_id(request.form['Movie_3']))
        liked.append(get_movie_id(request.form['Movie_4']))
        liked.append(get_movie_id(request.form['Movie_5']))
        recomen = reco(liked)
        print(liked,recomen)
        liked_posters = GetPoster(liked)
        reco_posters = GetPoster(recomen)
        return render_template('Reco.html',movie1 =liked_posters[0],movie2 =liked_posters[1],\
                            movie3 =liked_posters[2],movie4 =liked_posters[3],movie5=liked_posters[4],\
                            movie6 =reco_posters[0],movie7 =reco_posters[1],\
                            movie8 =reco_posters[2],movie9 =reco_posters[3],movie10=reco_posters[4])
#@app.route('/recomap',methods=['GET'])
#def recomap():
#    if request.method == 'GET':
#        liked_posters = GetPoster(app.liked)
#        reco_posters = GetPoster(app.reco)
#        return render_template('Reco.html',movie1 =liked_posters[0],movie2 =liked_posters[1],\
#                            movie3 =liked_posters[2],movie4 =liked_posters[3],movie5=liked_posters[4],\
#                            movie6 =reco_posters[0],movie7 =reco_posters[1],\
#                            movie8 =reco_posters[2],movie9 =reco_posters[3],movie10=reco_posters[4])

@app.route('/MovieDetector',methods=['GET','POST'])
def MovieDetector():
    if request.method == 'GET':
        return render_template('Movies-Detector.html')
    if request.method =='POST':
        inspect  = get_movie_id(request.form['Movie'])
        with open('files/inspect.txt','w+') as fin:
            fin.write(inspect)
        del fin
        return redirect(url_for('MovieDetail'))

@app.route('/MovieDetail',methods=['GET','POST'])
def MovieDetail():
        with open('files/inspect.txt','r') as fin:
            inspect = fin.read()
        print(inspect)
        del fin
        details = return_detail(inspect)
        l = len(details['Actors'])
        if l<3:
            details['Actors'] += ['']*(3-l)
        details['poster'] = " background-image: url(" + details['poster'] + ");min-height: 453px; background-position: 50% 50%"
#        details['hist'].output_backend = "svg"
#        export_svgs(details['hist'], filename="static/images/plot.svg")
        #export_png(details['hist'], filename="static/images/plot.png")
#        script,div = components(details['hist'])
        return render_template('Movies-body.html',Title=details['Title'],style=details['poster'],\
                                Top1000 = details['Top1000'], NonUS=details['NonUS'], Genre=details['Genre'],\
                                Director=details['Director'],IMDb=details['imdbRating'],Actor3=details['Actors'][0],\
                                Actor1=details['Actors'][1],Actor2=details['Actors'][2])



if __name__ == '__main__':
    app.run(debug=True)
