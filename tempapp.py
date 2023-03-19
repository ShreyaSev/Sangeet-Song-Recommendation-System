import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from collections import defaultdict

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="insert_your_client_id", client_secret="insert_your_client_Secret"))


def load_data():
    data = pd.read_csv("data/data.csv")
    tempdf={'valence':0,'year':0,'acousticness':0,'artists':'','danceability':0,'duration_ms':0,'energy':0,'explicit':0,'id':'','instrumentalness':0,'key':0,'liveliness':0,'loudness':0,'mode':0,'name':' ','popularity':0,'release_date':0,'speechiness':0,'tempo':0}
    data.append(tempdf,ignore_index=True)
    data['year_str']=data['year'].astype({'year':str})
    
    data['detail']=data['artists'] +'\n'+ data['year_str']
    
    return data

def find_song(name, year):
    
    song_data = defaultdict()
    
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    
    try:
        #getting the song name and corresponding year. 
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        #gets the information of the songs by condition (entire row of information about the song)
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    
    for key in dict_list[0].keys():
        #keys here are "name" and "year" of song
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=20):

    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)
    X = spotify_data.select_dtypes(np.number)
    number_cols = list(X.columns)
    #song_cluster_pipeline.fit(X)
    #song_cluster_labels = song_cluster_pipeline.predict(X)
    #spotify_data['cluster_label'] = song_cluster_labels
    
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


    metadata_cols = ['name', 'year', 'artists']
    
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
   
    scaler = StandardScaler()

    scaler.fit(X)

    scaled_data = scaler.transform(spotify_data[number_cols])
    
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]

    
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    return rec_songs[metadata_cols].to_dict(orient='records')


def display_songs(input_data):

    for song in input_data:
        for name,artists,year in song:
            artists = artists.strip('][').split(',')

            st.write(name)
            st.write(artists)
            st.write(year)

def songs(song_list,data):
    for i in range(len(song_list)):
        input=data[data['name']==song_list[i]]
        input=input.iloc[:,[14,3,1]]
        st.dataframe(input)
        input=np.array(input)
        #print(input[0,0])


def main():
    st.title('Sangeet- Song recommendation system')

    menu=['Home','Recommend']
    #menu={'Home':'Home','Recommend':'Recommend','About':'About'}
    choice= st.sidebar.selectbox("Menu",menu)
    #choice=option_menu(menu_title='Navigate',options=menu,default_index=0,orientation='horizontal')



    data=load_data()

    if choice=='Home':
        st.subheader("Home")
        st.text('Our song rec system')

    elif choice=='Recommend':

        k=0
        k1=0
        st.subheader("Recommend Songs")
        num_of_rec=st.sidebar.number_input("Number",2,7,5)

        playlist_name=st.sidebar.text_input('Playlist Name')

        song_list=[]

        def song_string(search_term):
            regex=search_term
            details=data[data['name'].str.match(regex)]
            details=details.iloc[:,[14,3,1]]
            #print(details)
            detail_dict={}
            for i in details.index:
                artist = details['artists'][i].strip('][').split(',')
                name=details['name'][i]
                year=details['year'][i]
                detail_dict[id]={'Name':name,'Artists':artist,'Year':year}

            #print(detail_dict)
            return detail_dict

        song_rec=[]

        for i in range(num_of_rec):

            temp={}

            with st.form(key='searchform'+str(i)):

                nav1,nav4,nav2,nav3=st.columns([5,2,5,2])
                songname=''
                
                with nav1:
                    search_term=st.selectbox(f"Song {i+1}",data.name.unique(),key=k)
                    songname+=search_term
                    #print(songname)
                    song_list.append(search_term)
                    k+=1
                    temp['name']=songname
                with nav4:
                    submit_search=st.form_submit_button('ok')
                with nav2:
                    d1=data[data['name']==songname]
                    #print(d1)
                    search_term=st.selectbox(f"Artists & Year",d1['detail'].unique(),key=k)
                    k+=1
                    #d2=d1[d1['detail']==search_term]
                    temp['year']=int(search_term[-4:])

                with nav3:
                    submit_search=st.form_submit_button()

            song_rec.append(temp)
        #print(song_rec)

        

        if st.button("Recommend"):
            if (search_term is not None):
                #print('recommended songs: ',recommend_songs(song_rec,data))

                final=pd.DataFrame(recommend_songs(song_rec,data))
                final["artists"]=final["artists"][1:-1]

                #st.dataframe(final)
                st.table(final)
                
                song_list=[]
                for i in final["name"]:
                    song_list.append(i)
                scope='playlist-modify-public'
                username='xr4j5rv64wwjaksq0x74gu63v'

                token=SpotifyOAuth(scope=scope,username=username,client_id="insert_your_client_id", client_secret="insert_your_client_secret",redirect_uri='insert_your_redirect_uri')
                spotify=spotipy.Spotify(auth_manager=token)

                #playlist_name="My Playlist"#the variable which you need to get from user
                print(playlist_name)
                spotify.user_playlist_create(user=username,name=playlist_name,public=True)
                list_of_songs=[]

                for i in range (len(song_list)):

                    results=spotify.search(q=song_list[i])
                    list_of_songs.append(results['tracks']['items'][0]['uri'])
                    
                    prePlaylists = spotify.user_playlists(user=username)
                    playlist=prePlaylists['items'][0]['id']
                    
                spotify.user_playlist_add_tracks(user=username,playlist_id=playlist,tracks=list_of_songs)


if __name__ == '__main__':
    main()
