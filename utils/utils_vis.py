# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:30:57 2021

@author: wangy79 change
"""
# from bs4 import BeautifulSoup
# from IPython.display import IFrame
import numpy as np
# import gmplot 
import matplotlib.pyplot as plt
import utils
from matplotlib.ticker import FormatStrFormatter
import math
from matplotlib import cm

# # for visualization
# def insertapikey(fname):
#     apikey = 'AIzaSyDBo88RY_39Evn87johzUvFw5x_Yg6cfkI'
# #      """put the google api key in a html file"""
#     print('\n###############################')
#     print('\n Beginning Key Insertion ...')

#     def putkey(htmltxt, apikey, apistring=None):
# #          """put the apikey in the htmltxt and return soup"""
#         if not apistring:
#             apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initialize&libraries=visualization&sensor=true_or_false"
#         soup = BeautifulSoup(htmltxt, 'html.parser')
#         soup.script.decompose() #remove the existing script tag
#         body = soup.body
#         src = apistring % (apikey, )
#         tscript = soup.new_tag("script", src=src) #, async="defer"
#         body.insert(-1, tscript)
#         return soup

#     htmltxt = open(fname, 'r').read()
#     # htmltxt = open(fname,'r+').read()
#     soup = putkey(htmltxt, apikey)
#     newtxt = soup.prettify()
#     open(fname, 'w').write(newtxt)
#     print('\nKey Insertion Completed!!')


# def jupyter_display(gmplot_filename):
#     google_api_key = 'AIzaSyDBo88RY_39Evn87johzUvFw5x_Yg6cfkI'
    
# #      """Hack to display a gmplot map in Jupyter"""
#     with open(gmplot_filename, "r+b") as f:
#         f_string = f.read()
#         url_pattern = "https://maps.googleapis.com/maps/api/js?libraries=visualization&sensor=true_or_false"
#         newstring = url_pattern + "&key=%s" % google_api_key
#         f_string = f_string.replace(url_pattern.encode(), newstring.encode())
#         f.write(f_string)
#     return IFrame(gmplot_filename, width=900, height=600)

# def draw_map_scatter(x,y):
    
#     map_name = "test.html"
#     gmap = gmplot.GoogleMapPlotter(x[0], y[0], 100) 

#     gmap.scatter(x, y, s=.9, alpha=.8, c='red',marker = False)
#     gmap.draw(map_name)
    
#     insertapikey(map_name)
#     return jupyter_display(map_name)
    
# def draw_map(df, latcenter, loncenter, nO):
    
#     map_name = "test.html"
#     gmap = gmplot.GoogleMapPlotter(latcenter, loncenter, 100) 

#     groups = df.groupby('ID')
#     groupList = list(groups.groups)

#     for i in groupList[:nO]:   
#         group = groups.get_group(i)
#         gmap.scatter(group.lat, group.lon, s=.5, alpha=.8, label=group.loc[group.index[0],'ID'],marker = False)
#     gmap.draw(map_name)
    
#     insertapikey(map_name)
#     return jupyter_display(map_name)

# # draw rectangles from 3D box on map
# def draw_map_box(Y, nO, lats, lngs):
    
#     map_name = "test.html"
#     notNan = ~np.isnan(np.sum(Y,axis=-1))
#     Y = Y[notNan,:]
#     gmap = gmplot.GoogleMapPlotter(Y[0,0], Y[0,1], nO) 

#     # get the bottom 4 points gps coords
#     # Y = np.array(df[['bbrlat','bbrlon','fbrlat','fbrlon','fbllat','fbllon','bbllat','bbllon']])
    

#     for i in range(len(Y)):
#         coord = Y[i,:]
#         coord = np.reshape(coord,(-1,2)).tolist()
#         coord.append(coord[0]) #repeat the first point to create a 'closed loop'
#         coord_tuple = [tuple(pt) for pt in coord]
#         rectangle = zip(*coord_tuple) #create lists of x and y values
#         gmap.polygon(*rectangle)    
#     lats = lats[~np.isnan(lats)]
#     lngs = lngs[~np.isnan(lngs)]
#     gmap.scatter(lats, lngs, color='red', size=1, marker=True)
#     gmap.scatter(Y[:,2], Y[:,3],color='red', size=0.1, marker=False)

#     gmap.draw(map_name)

#     insertapikey(map_name)
#     return jupyter_display(map_name)
    
# def plot_frame(x, y, x_id, y_id, xmin, xmax, frame_id = 0, length=15,width=4):
#     fig, ax = plt.subplots(figsize=(length,width))

#     for i in range(len(x)):
#         coord = x[i,:]
#         coord = np.reshape(coord,(-1,2)).tolist()
#         coord.append(coord[0]) #repeat the first point to create a 'closed loop'
#         xs, ys = zip(*coord) #lon, lat as x, y
#         plt.plot(xs,ys, c='r', label='pred' if i==0 else '')#alpha=i/len(D)
#         plt.text(xs[0], ys[0], str(x_id[i]), fontsize=8)
#         plt.scatter(x[i,2],x[i,3],color='r') # 

#     for i in range(len(y)):
#         coord = y[i,:]
#         coord = np.reshape(coord,(-1,2)).tolist()
#         coord.append(coord[0]) #repeat the first point to create a 'closed loop'
#         xs, ys = zip(*coord) #lon, lat as x, y
#         plt.plot(xs,ys, c='b', label='meas' if i==0 else '')#alpha=i/len(D)
#         plt.text(xs[0], ys[0], str(y_id[i]), fontsize=8)
#         plt.scatter(y[i,2],y[i,3],color='b') # 
        
#     plt.xlabel('meter')
#     plt.ylabel('meter')
    
#     plt.xlim([xmin,xmax])
#     plt.ylim([0,45])
#     plt.title(frame_id)
#     plt.legend()
#     plt.show() 
#     return

def plot_track(x):
    fig, ax = plt.subplots(figsize=(15,1))

    for i in range(len(x)):
        coord = x[i,:]
        coord = np.reshape(coord,(-1,2)).tolist()
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        xs, ys = zip(*coord) #lon, lat as x, y
        plt.plot(xs,ys, c='r', label='pred' if i==0 else '')#alpha=i/len(D)
        # plt.text(xs[0], ys[0], str(x_id[i]), fontsize=8)
        plt.scatter(x[i,2],x[i,3],color='r') # 

    
    plt.xlabel('meter')
    plt.ylabel('meter')
    
    # plt.xlim([xmin,xmax])
    # plt.ylim([0,45])
    # plt.title(frame_id)
    plt.legend()
    plt.show() 
    return
    
def plot_track_df(df,length=15,width=1,legend = "rectified", show=True, ax=None, color='black',title=""):
    carid = df["ID"].iloc[0]
    D = np.array(df[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
    if ax is None:
        fig, ax = plt.subplots(figsize=(length,width))
    
    for i in range(len(D)):
        coord = D[i,:]
        coord = np.reshape(coord,(-1,2)).tolist()
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        xs, ys = zip(*coord) #lon, lat as x, y
        ax.plot(xs,ys,label=legend if i==0 else '',c=color)
        ax.scatter(D[i,2],D[i,3],color=color)#,alpha=i/len(D)
    ax.set_xlabel('meter')
    ax.set_ylabel('meter')
    ax.set_title(title)
    ax.legend()
    if show:
        plt.show() 
        return
    else:
        return ax
    

def plot_track_df_camera(df,tform_path,length=15,width=1, camera='varies'):
    camera_list = ['p1c1','p1c2','p1c3','p1c4','p1c5','p1c6']
    color=cm.rainbow(np.linspace(0,1,len(camera_list)))
    camera_dict = dict(zip(camera_list,color))
    ID = df['ID'].iloc[0]
    fig, ax = plt.subplots(figsize=(length,width))
    if camera == 'varies':
        camera_group = df.groupby('camera')
        print('ID:',ID,'# frames:',len(df),'# cameras:',len(camera_group))
        for cameraID,cg in camera_group:
            Y = np.array(cg[['bbr_x','bbr_y', 'fbr_x','fbr_y','fbl_x','fbl_y','bbl_x', 'bbl_y']])
            c=camera_dict[cameraID]
            for i in range(len(Y)):
                coord = Y[i,:]
                coord = np.reshape(coord,(-1,2)).tolist()
                coord.append(coord[0]) #repeat the first point to create a 'closed loop'
                xs, ys = zip(*coord) #lon, lat as x, y     
                plt.plot(xs,ys,c=c,label=cameraID if i == 0 else "")

            plt.scatter(Y[:,2],Y[:,3],color='black')
            ax = plt.gca()
            plt.xlabel('meter')
            plt.ylabel('meter')
            plt.legend()
            ax.format_coord = lambda x,y: '%.6f, %.6f' % (x,y) 
        
    else:
        c=camera_dict[camera]
        img_pts = np.array(df[['bbrx','bbry', 'fbrx','fbry','fblx','fbly','bblx', 'bbly']])
        Y = utils.img_to_road_box(img_pts,tform_path,camera)
        for i in range(len(Y)):
            coord = Y[i,:]
            coord = np.reshape(coord,(-1,2)).tolist()
            coord.append(coord[0]) #repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord) #lon, lat as x, y     
            plt.plot(xs,ys,c=c,label=camera if i == 0 else "")

        plt.scatter(Y[:,2],Y[:,3],color='black')
        ax = plt.gca()
        plt.xlabel('meter')
        plt.ylabel('meter')
        ax.format_coord = lambda x,y: '%.6f, %.6f' % (x,y) 

        plt.legend()
        plt.show()
    
    return

def plot_track_compare(car,carre, legends=None):
    ax = plot_track_df(car,legend = legends[0], show=False, color='red')
    ax = plot_track_df(carre, legend = legends[1], show=False, ax=ax, color='blue')
    return

def plot_lane_distribution(df):
# plot lane distribution
    plt.figure()
    df = utils.assign_lane(df)
        
    width = 0.3
    x1 = df.groupby('lane').ID.nunique() # count unique IDs in each lane
    plt.bar(x1.index-0.1,x1.values,color = "r",width = width)
        
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Lane index')
    plt.ylabel('ID count')
    plt.title('Lane distribution')
    return

def plot_time_space(df, lanes=[1], time="frame", space="x", ax=None, show =True):
        
    # plot time space diagram (4 lanes +1 direction)
    if ax is None:
        fig, ax = plt.subplots()
    
    
    colors = ["blue","orange","green","red","purple"]
    for i,lane_idx in enumerate(lanes):
        lane = df[df['lane']==lane_idx]
        groups = lane.groupby('ID')
        j = 0
        for carid, group in groups:
            if time=="frame":
                x = group['Frame #'].values
            else:
                x = group['Timestamp'].values
            if space == "x":
                y1 = group['bbr_x'].values
                y2 = group['fbr_x'].values
            else:
                y1 = group['bbr_y'].values
                y2 = group['bbl_y'].values
            if len(lanes)>1:
                ax.fill_between(x,y1,y2,alpha=0.5,color = colors[j%len(colors)], label="lane {}".format(lane_idx) if j==0 else "")
            else:
                ax.fill_between(x,y1,y2,alpha=0.5,color = colors[j%len(colors)], label="{}".format(carid))
            j += 1
        try:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        except:
            pass
        if time=="frame":
            ax.set_xlabel('Frame')
        else:
            ax.set_xlabel('Time')
        ax.set_ylabel('x (m)' if space=="x" else 'y(m)')
        ax.set_title('Lane {}'.format(lane_idx)) 
    return None if show else ax


def dashboard(cars, states = None, car_legends=None):
    '''
    cars: list of dfs
    states: list of state names to be plotted
        if None, plot acceleration/speed/theta/... of each car
    car_legends: list of names to put as legends
    '''
    if car_legends:
        assert len(cars) == len(car_legends)
    n_states = len(states)
    fig, axs = plt.subplots(math.ceil(n_states/3), 3, figsize=(18,18))
    axs = axs.ravel()
    
    colors = ["blue","orange","green","red","purple"]
    units = {"x": "(m)",
             "speed": "(m/s)",
             "acceleration": "(m/s2)",
             "jerk": "(m/s3)",
             "y": "(m)",
             "theta": "rad"
             }
    
    # carid = cars[0]["ID"].iloc[0]
    meas_states = {"x","y"}
    for stateidx, state in enumerate(states):
        
        for caridx, car in enumerate(cars):
            if car_legends[caridx] not in {'rectified','gt','meas'}:
                car = utils.calc_dynamics_car(car)
                print(car_legends[caridx], ' calculate dynamics')
             
            frames = car['Frame #'].values
            c = colors[caridx%len(colors)]
            
            # time vs. state
            if car_legends[caridx]=="meas" and state not in meas_states:
                continue
            state_values = car[state].values
            axs[stateidx].scatter(frames, state_values,color=c,s=2,label="{}".format(car_legends[caridx]))
            axs[stateidx].plot(frames, state_values,color=c)
            
        axs[stateidx].legend()            
        axs[stateidx].set_xlabel('Frame #')
        # axs[stateidx].set_ylabel(state + " " + units[state])
        try:
            axs[stateidx].set_title(state + " " + units[state])  
        except:
            axs[stateidx].set_title(state)  
    plt.show()
    return
   
    
   
def plot_histogram(data_list, bins, labels, xlabel, ylabel, title):
    '''
    plot a list of data together on the same plot
    data_list: a list of lists or np array
    bins: histogram bins
    labels: data label, same length as data_list
    xlabel:
    ylabel
    title
    '''
    # remove nans
    
    color = ["r","g","b"]
    fig, ax1 = plt.subplots(1, 1)
    if isinstance(data_list, list):
        for data in data_list:
            data = data[~np.isnan(data)]
        bs=np.histogram(np.hstack((data_list)), bins=bins)[1]
        bw = bs[1]-bs[0]
        for i,data in enumerate(data_list):
            ax1.hist(data, bins = bs, density = True, weights = [bw]*len(data), facecolor=color[i%len(color)], alpha=0.75, label=labels[i])
    else:
        data_list = np.fromiter(data_list, dtype=float)
        data_list = data_list[~np.isnan(data_list)]
        bs=np.histogram(data_list, bins=bins)[1]
        bw = bs[1]-bs[0]
        ax1.hist(data_list, bins = bs, density = True, weights = [bw]*len(data_list), facecolor=color[0], alpha=0.75, label=labels)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.grid()
    ax1.legend()
    return
