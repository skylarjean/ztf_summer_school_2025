import matplotlib.pyplot as plt
from matplotlib import colormaps as cmap
from matplotlib.patches import Rectangle, Circle, Wedge, ConnectionPatch ,FancyArrowPatch
import numpy as np
import json

def wedge_with_text(ax,xy,radius,t1,t2,text,edgecolor='k',fc='r',fontsize=5,lw=2,text_color='k',alpha=1,skip_text=False,**kwargs):
    circ=Wedge(xy,radius,t1,t2,edgecolor=edgecolor,fc=fc,lw=lw,alpha=alpha)
    ax.add_patch(circ)
    if skip_text:
        return circ
    ax.annotate(text, (xy[0], xy[1]), color=text_color, weight='bold', 
                fontsize=fontsize, ha='center', va='center')
    return circ

default_patch_kwargs={'edgecolor':'k','fc':'r','lw':2}
default_text_kwargs= {'fontsize':10,'weight':'bold','ha':'center','va':'center'}
import copy

def split_circle(ax,center,radius,w1_text='w1',w2_text='w2',title_text='title',w1_kwargs={},w2_kwargs={},text_kwargs={},skip_text=False):
    
    w1_kwargs_internal=copy.deepcopy(default_patch_kwargs)
    w2_kwargs_internal=copy.deepcopy(default_patch_kwargs)
    text_kwargs_internal=copy.deepcopy(default_text_kwargs)

    w1_kwargs_internal.update(w1_kwargs)
    w2_kwargs_internal.update(w2_kwargs)
    text_kwargs_internal.update(text_kwargs)

    #make the circle
    w1=Wedge(center,radius,0,180,**w1_kwargs_internal)
    w2=Wedge(center,radius,180,0,**w2_kwargs_internal)
    ax.add_patch(w1)
    ax.add_patch(w2)
    if skip_text:
        return
    # add text
    ax.annotate(w1_text,(center[0],center[1]+radius/2),**text_kwargs_internal)
    ax.annotate(w2_text,(center[0],center[1]-radius/2),**text_kwargs_internal)
    ax.annotate(title_text,(center[0],center[1]+radius*1.2),**text_kwargs_internal)

def update_tax(dict,df):
    key=dict['key']
    if f'{key}_dnn' in df.keys():
        dict['dnn_value']=df[f'{key}_dnn'][0]
    else:
        dict['dnn_value']=0
    if f'{key}_xgb' in df.keys():
        dict['xgb_value']=df[f'{key}_xgb'][0]
    else:
        dict['xgb_value']=0
    if 'children' not in dict.keys(): #hit the bottom
        return
    for subdict in dict['children']:
        update_tax(subdict,df)
def tree_info(sub_tree,depth=0,depth_guide=None):
    depth_guide[depth]+=len(sub_tree) #add the curent number of nodes
    for leaf in sub_tree:
        if 'children' not in leaf.keys():
            continue
        else: #got deeper
            tree_info(leaf['children'],depth=depth+1,depth_guide=depth_guide)
def get_tree_info(tree):
    depth_guide=np.zeros(100).astype(int)
    tree_info([tree],depth_guide=depth_guide)
    depth_guide=depth_guide[depth_guide!=0]
    return depth_guide

def all_centers(depth_guide,sep=.1):
    max_depth=len(depth_guide)
    master_list=[]
    flat_list=[]
    for depth,nodes_in_depth in enumerate(depth_guide):
        center_list=[]
        for i in range(nodes_in_depth):
            x_cen=(depth+1)/(max_depth+1)
            y_cen=(i+1)/(nodes_in_depth+1)
            radius=min([0.5/max_depth,0.5/nodes_in_depth])*(1-sep)
            center_list.append([x_cen,y_cen,radius])
            flat_list.append([x_cen,y_cen,radius])
        master_list.append(center_list)
    return master_list,flat_list

def add_plot_info_to_tree(tree_list,param_list,depth=0): #D: only is writing to first few indecis
    for i,sub_tree in enumerate(tree_list):
        loop_info=param_list[depth].pop()
        sub_tree['center_x']=loop_info[0]
        sub_tree['center_y']=loop_info[1]
        sub_tree['radius']=loop_info[2]
        if 'children' in sub_tree.keys():
            #next level
            add_plot_info_to_tree(sub_tree['children'],param_list,depth=depth+1)
default_recur_split_circle_kwargs={'radius':.2,
'baseline_x':.25,'baseline_y':1,
'base_x':0,'base_y':0,
'sep':0.1,
'cm':'Blues',
'skip_text':False,
'draw_lines':False,
'parent_center':(0,0)}

def recur_split_circles(ax,children,settings={}):
    k=copy.deepcopy(default_recur_split_circle_kwargs)
    k.update(settings)
    #draw current rectangle
    for i,class_dict in enumerate(children):
        radius=class_dict['radius']
        center_x=class_dict['center_x']
        center_y=class_dict['center_y']
        w1_kwargs={'fc':cmap.get_cmap(k['cm'])(class_dict['dnn_value']/1.3),'lw':min(2,20*radius)}
        w2_kwargs={'fc':cmap.get_cmap(k['cm'])(class_dict['xgb_value']/1.3),'lw':min(2,20*radius)}
        text_kwargs={'fontsize':min(9,300*radius)}
        split_circle(ax,(center_x,center_y),radius,title_text=class_dict['name'],
                                                        w1_kwargs=w1_kwargs, w1_text=f"{class_dict['dnn_value']:1.2f}",
                                                        w2_kwargs=w2_kwargs, w2_text=f"{class_dict['xgb_value']:1.2f}",
                                                        text_kwargs=text_kwargs,
                                                        skip_text=k['skip_text'])
        if k['draw_lines']:
            ax.add_patch(FancyArrowPatch(k['parent_center'],(center_x,center_y),zorder=-100,lw=1,alpha=.5))
        #draw circle of radius r at position base_x, base_y+(R*(1+sep)+i*2R(1+sep))
        if ('children' in class_dict.keys()) and len(class_dict['children'])!=0:
            #compute new r and base_x base_y
            new_data={'draw_lines':True,'parent_center':(center_x,center_y)}
            new_settings=copy.deepcopy(k)
            new_settings.update(new_data)
            recur_split_circles(ax,class_dict['children'],settings=new_settings)
    pass

def plot_classifications(ax,sample,tree,sep=.1,settings={}):
    update_tax(tree,sample)
    dg=get_tree_info(tree)
    master_list,flat_list=all_centers(dg,sep=sep)
    #flat_list=np.array(flat_list)
    #ax.scatter(flat_list[:,0],flat_list[:,1],color='k',zorder=1000)
    add_plot_info_to_tree([tree],master_list)
    recur_split_circles(ax,[tree],settings=settings)
    
    
if __name__=="__main__":
    import pandas as pd
    import yaml
    import copy
    data=pd.read_csv('/Users/danielwarshofsky/Isolated/similarity_search/1mill_SCoPe_subsample.csv')
    sample=data.head(1)
    full_tree=None
    with open('/Users/danielwarshofsky/Isolated/all_tax.yaml') as config_yaml:
            full_tree = yaml.load(config_yaml, Loader=yaml.FullLoader)
    d=np.random.uniform(0,1,size=len(sample.columns))
    df=pd.DataFrame(columns=sample.columns)
    df.loc[0]=d
    copy_tree=copy.deepcopy(full_tree)
    ph_tree=copy_tree['children'][0]
    on_tree=copy_tree['children'][1]
    fig,axs=plt.subplots(2,figsize=(8,16))

    s_ph={'skip_text':False,'cm':"Greens"}
    s_on={'skip_text':False,'cm':"Greens"}
    axs[0].set_title('Phenomenological')
    plot_classifications(axs[0],df,ph_tree,sep=.3,settings=s_ph)
    axs[1].set_title('Ontological')
    plot_classifications(axs[1],df,on_tree,sep=.3,settings=s_on)
    #ax.vlines([.25,.5,.75,1],0,1)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.tight_layout()