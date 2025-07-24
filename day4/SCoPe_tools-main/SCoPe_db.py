from penquins import Kowalski
import pandas as pd
import os
import yaml
class scope_client():
    def __init__(self,config='config.yaml', time_out=120,max_n_threads=4):

        assert os.path.exists(config), f"{config} is not a valid path"

        with open('config.yaml','r') as f:
            self.config_settings=yaml.load(f, Loader=yaml.FullLoader)
        self.max_n_threads=max_n_threads
        self._setup_kowalski_(self.config_settings['hosts'],time_out)

        self.features_keys=self.config_settings['features_keys']
        self.classification_keys=self.config_settings['classification_keys']
        self._setup_projections_()       
    def _setup_kowalski_(self,hosts: dict, time_out: int):
        instances = {
            host: {
                'protocol': 'https',
                'port': 443,
                'host': f'{host}.caltech.edu',
                'username': hosts[host]['username'],
                'password': hosts[host]['password']
            }
            for host in hosts.keys()
        }
        self.kowalski_instances = Kowalski(timeout=time_out, instances=instances)
    def _setup_projections_(self):
        self.projection_features={key:1 for key in self.features_keys}
        self.projection_classification={key:1 for key in self.classification_keys}   
    def cone_search(self,ra,dec,radius=2,unit='arcsec') -> pd.DataFrame:
        #cone search
        assert unit in ['deg','arcmin','arcsec'],'Unit must be "deg","arcmin" or "arcsec"'
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": radius,
                    "cone_search_unit": unit,
                    "radec": {
                        "center": [
                            ra,
                            dec
                        ]
                    }
                },
                "catalogs": {
                    "ZTF_source_features_DR16": {
                        'projection':self.projection_features
                    },
                    "ZTF_source_classifications_DR16":{
                        'projection':self.projection_classification
                    }
                    }
            },
            "kwargs": {
                "filter_first": False
            }
        }

        response = self.kowalski_instances.query(query=query).get("gloria")
        if response.get('status') != 'success':
            print('Querey failed')
            print(response.get("message"))
            return None
        #check for valid response
        
        df_f=pd.DataFrame(response.get("data").get("ZTF_source_features_DR16").get('center'))
        df_c=pd.DataFrame(response.get("data").get("ZTF_source_classifications_DR16").get('center'))
        if len(df_c)==0 or len(df_f)==0:
            print('No sources found')
            return None
        df=df_f.merge(df_c,on='_id')
        return df
    def cone_searches(self,positions: list, radius=2, unit='arcsec') -> pd.DataFrame:
        assert unit in ['deg','arcmin','arcsec'],'Unit must be "deg","arcmin" or "arcsec"'
        if not isinstance(positions,list):
            print("positions must be a list")
            return None
        ra_decs={f'{i}':[positions[i][0],positions[i][1]]for i in range(len(positions))}
        query= {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": radius,
                    "cone_search_unit": unit,
                    "radec": ra_decs
                },
                "catalogs": {
                    "ZTF_source_features_DR16": {
                        'projection':self.projection_features
                    },
                    "ZTF_source_classifications_DR16":{
                        'projection':self.projection_classification
                    }
                    }
            },
            "kwargs": {
                "filter_first": False
            }
        }
        response = self.kowalski_instances.query(query=query).get("gloria")
        if response.get('status') != 'success':
            print('Querey failed')
            print(response.get("message"))
            return None

        classification_data=[]
        for key in response['data']['ZTF_source_classifications_DR16'].keys():
            classification_data+=response['data']['ZTF_source_classifications_DR16'][key]
        df_c=pd.DataFrame(classification_data)

        feature_data=[]
        for key in response['data']['ZTF_source_features_DR16'].keys():
            feature_data+=response['data']['ZTF_source_features_DR16'][key]
        df_f=pd.DataFrame(feature_data)
        if len(df_c)==0 or len(df_f)==0:
            print('No sources found')
            return None
        df=df_f.merge(df_c,on='_id')
        return df
    def ids_search(self,id,id_type='_id') -> pd.DataFrame:
        assert id_type in ["_id","AllWISE___id","Gaia_EDR3___id","PS1_DR1___id"], 'id_type must be in ["_id","AllWISE___id","Gaia_EDR3___id","PS1_DR1___id"]'
        #fist grab from features catalog, it has more ids
        #then collect on _id
        if not isinstance(id,list):
            id=[id]
        for i in range(len(id)):
            id[i]= int(id[i])
        match_id_f={id_type:{'$in':id}}
        pipeline_features=[
            {'$match': match_id_f},
            {'$project':self.projection_features},
        ]
        query_features = {
        "query_type": "aggregate",
        "query": {
            "catalog":"ZTF_source_features_DR16",
            "pipeline": pipeline_features
            }
        }
        response_features = self.kowalski_instances.query(query=query_features).get('gloria')
        if response_features.get('status') != 'success':
            print('Querey failed')
            print(response_features.get("message"))
            return None
        df_f=pd.DataFrame(response_features.get('data'))
        ztf_ids=[ztf_id for ztf_id in df_f['_id']]
        
        match_id_c={'_id':{'$in':ztf_ids}}
        pipeline_class=[
            {'$match': match_id_c},
            {'$project':self.projection_classification},
        ]
        query_class = {
        "query_type": "aggregate",
        "query": {
            "catalog":"ZTF_source_classifications_DR16",
            "pipeline": pipeline_class
            }
        }
        response_class = self.kowalski_instances.query(query=query_class).get('gloria')
        df_c=pd.DataFrame(response_class.get('data'))

        df=df_f.merge(df_c,on='_id')
        return df
    def _batch_over_fields_aggregate_(self,fields:list, stages: list, catalog: str):
        field_stages=[{'field':{'$in':[f]}} for f in fields]
        pipelines=[[{'$match':f}]+stages for f in field_stages]
        qs=[
            {"query_type": "aggregate",
                "query": {
                "catalog":catalog,
                "pipeline": pipeline
                }
            }
            for pipeline in pipelines
        ]
        response=self.kowalski_instances.query(queries=qs,use_batch_query=True,max_n_threads=self.max_n_threads)
        return response
    def search_by_classification(self,fields:list,filter_stage) -> pd.DataFrame:
        assert isinstance(fields,list) and len(fields)!=0, 'Fields must be a non-empty list'
        class_stages=[{'$match':filter_stage},{'$project':self.projection_classification}]
        class_response=self._batch_over_fields_aggregate_(fields,class_stages,'ZTF_source_classifications_DR16').get('gloria')
        #make df
        temp_list=[]
        for i,rc in enumerate(class_response):
            if rc.get('status') != 'success':
                print(f'Query failed for field {fields[i]} on ZTF_source_classifications_DR16')
                print(rc.get("message"))
            else:
                temp_list+=rc.get('data')
        df_c=pd.DataFrame(temp_list)
        del temp_list
        #collect ids
        ids=[_id for _id in df_c['_id']]
        match_id={
        '_id':{'$in':ids},
        }
        #ask of info from features
        feature_stages=[{'$match':match_id},{'$project':self.projection_features}]
        feature_response=self._batch_over_fields_aggregate_(fields,feature_stages,'ZTF_source_features_DR16').get('gloria')

        temp_list=[]
        for i,rf in enumerate(feature_response):
            if rf.get('status') != 'success':
                print(f'Query failed for field {fields[i]} on ZTF_source_features_DR16')
                print(rf.get("message"))
            else:
                temp_list+=rf.get('data')
        df_f=pd.DataFrame(temp_list)

        df=df_f.merge(df_c,on='_id')
        return df
    def search_by_feature(self,fields:list,filter_stage) -> pd.DataFrame:
        assert isinstance(fields,list) and len(fields)!=0, 'Fields must be a non-empty list'
        feature_stages=[{'$match':filter_stage},{'$project':self.projection_features}]
        feature_response=self._batch_over_fields_aggregate_(fields,feature_stages,'ZTF_source_features_DR16').get('gloria')
        #make df
        temp_list=[]
        for i,rf in enumerate(feature_response):
            if rf.get('status') != 'success':
                print(f'Query failed for field {fields[i]} on ZTF_source_features_DR16')
                print(rf.get("message"))
            else:
                temp_list+=rf.get('data')
        df_f=pd.DataFrame(temp_list)
        del temp_list
        #collect ids
        ids=[_id for _id in df_f['_id']]
        match_id={
        '_id':{'$in':ids},
        }
        #ask of info from features
        class_stages=[{'$match':match_id},{'$project':self.projection_classification}]
        class_response=self._batch_over_fields_aggregate_(fields,class_stages,'ZTF_source_classifications_DR16').get('gloria')

        temp_list=[]
        for i,rc in enumerate(class_response):
            if rc.get('status') != 'success':
                print(f'Query failed for field {fields[i]} on ZTF_source_classifications_DR16')
                print(rc.get("message"))
            else:
                temp_list+=rc.get('data')
        df_c=pd.DataFrame(temp_list)

        df=df_f.merge(df_c,on='_id')
        return df
    def get_indecies(self):
        #not implemented
        q_f={"query_type": "info",
            "query": {
            "catalog":"ZTF_source_features_DR16",
            "command": 'index_info'
            }
        }
        q_c={"query_type": "info",
            "query": {
            "catalog":"ZTF_source_classifications_DR16",
            "command": 'index_info'
            }
        }
        class_ind=self.kowalski_instances.query(query=q_c).get('gloria').get('data')
        feature_ind=self.kowalski_instances.query(query=q_f).get('gloria').get('data')
        return class_ind,feature_ind
    def get_light_curves_by_id(self,ids:list,catalog='ZTF_sources_20240117') -> pd.DataFrame:
        assert isinstance(ids,list),'Ids must be a list'
        for i in range(len(ids)):
            ids[i]= int(ids[i])
        match_id_f={'_id':{'$in':ids}}
        pipeline_features=[
            {'$match': match_id_f},
        ]
        q= {
        "query_type": "aggregate",
        "query": {
            "catalog":catalog,
            "pipeline": pipeline_features
            }
        }
        response = self.kowalski_instances.query(query=q).get('melman')
        if response.get('status') != 'success':
            print('Querey failed')
            print(response.get("message"))
            return None
        df=pd.DataFrame()
        for lc in response.get('data'):
            loop_df=pd.DataFrame(lc.get('data'))
            loop_df['_id']=lc['_id']
            df=pd.concat([df,loop_df],ignore_index=True)
        df.sort_values(by='hjd')
        return df
    def get_light_curves_by_coord(self,ra,dec,catalog='ZTF_sources_20240117') -> pd.DataFrame:
        q= {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": 2,
                    "cone_search_unit": 'arcsec',
                    "radec": {
                        "center": [
                            ra,
                            dec
                        ]
                    }
                },
                "catalogs": {
                    catalog: {}
                }
            }
        }
        response = self.kowalski_instances.query(query=q).get('melman')
        if response.get('status') != 'success':
            print('Querey failed')
            print(response.get("message"))
            return None
        df=pd.DataFrame()
        for lc in response.get('data').get(catalog).get('center'):
            loop_df=pd.DataFrame(lc.get('data'))
            loop_df['_id']=lc['_id']
            df=pd.concat([df,loop_df],ignore_index=True)
        df.sort_values(by='hjd')
        return df