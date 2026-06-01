# import netCDF4
# from netCDF4 import Dataset
# import numpy as np
# import pandas as pd
# from torch.autograd import Variable
# import torch
# from haversine import haversine_vector, Unit
# from pylab import mpl
#
# mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS ']
#
#
# class LoadData():
#     def __init__(self, train_ratio, valid_ratio, input_seq_len, output_seq_len):
#         self.X, self.Y, self.num, self.num_nodes, self.features, self.coordinates, self.mete_long, self.mete_lat = self._load()
#         self.train_ratio = train_ratio
#         self.valid_ratio = valid_ratio
#         self.input_seq_len = input_seq_len
#         self.output_seq_len = output_seq_len
#         self._split(int(self.train_ratio * self.num), int((self.valid_ratio + self.train_ratio) * self.num))
#         self.adj = self._getAdj()
#
#     def get_azimuth(self):
#         coords = self.coordinates  # numpy: (N,2) -> [lat, lon]
#         N = self.num_nodes
#         az = torch.zeros((N, N), dtype=torch.long)
#
#         code_map = [5, 3, 8, 2, 6, 4, 7, 1]
#
#         for i in range(N):
#             lat_i, lon_i = coords[i, 0], coords[i, 1]
#             for j in range(N):
#                 if i == j:
#                     continue
#                 lat_j, lon_j = coords[j, 0], coords[j, 1]
#                 dy = lat_j - lat_i
#                 dx = lon_j - lon_i
#                 if dx == 0 and dy == 0:
#                     continue
#                 angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
#                 sector = int(((angle + 22.5) // 45) % 8)  # 0..7
#                 az[i, j] = code_map[sector]
#
#         return az
#
#     def dimensionlessProcessing(self, df_values):
#         from sklearn.preprocessing import StandardScaler
#         scaler = StandardScaler()
#         res = scaler.fit_transform(df_values)
#         return pd.DataFrame(res)
#
#     def _load(self):
#         for i in range(1, 13):
#
#             nc_obj_2023 = Dataset(
#                 '/tmp/pycharm_project_613/dataset/CSJ_91stations/processed_meteroloy_data_ws+wd/m20230' + str(
#                     i) + '.nc')
#             if i == 1:
#                 longitude_2023 = np.array((nc_obj_2023.variables['longitude'][:]))
#                 latitude_2023 = np.array((nc_obj_2023.variables['latitude'][:]))
#                 time_2023 = np.array((nc_obj_2023.variables['valid_time'][:]))
#                 t2m_2023 = np.array((nc_obj_2023.variables['t2m'][:]))
#                 ws_2023 = np.array((nc_obj_2023.variables['ws'][:]))
#                 wd_2023 = np.array((nc_obj_2023.variables['wd'][:]))
#                 d2m_2023 = np.array((nc_obj_2023.variables['d2m'][:]))
#                 sp_2023 = np.array((nc_obj_2023.variables['sp'][:]))
#                 tp_2023 = np.array((nc_obj_2023.variables['tp'][:]))
#             else:
#                 time_2023 = np.concatenate((time_2023, np.array((nc_obj_2023.variables['valid_time'][:]))), axis=0)
#                 t2m_2023 = np.concatenate((t2m_2023, np.array((nc_obj_2023.variables['t2m'][:]))), axis=0)
#                 ws_2023 = np.concatenate((ws_2023, np.array((nc_obj_2023.variables['ws'][:]))), axis=0)
#                 wd_2023 = np.concatenate((wd_2023, np.array((nc_obj_2023.variables['wd'][:]))), axis=0)
#                 d2m_2023 = np.concatenate((d2m_2023, np.array((nc_obj_2023.variables['d2m'][:]))), axis=0)
#                 sp_2023 = np.concatenate((sp_2023, np.array((nc_obj_2023.variables['sp'][:]))), axis=0)
#                 tp_2023 = np.concatenate((tp_2023, np.array((nc_obj_2023.variables['tp'][:]))), axis=0)
#
#         for i in range(1, 13):
#             nc_obj = Dataset(
#                 '/tmp/pycharm_project_613/dataset/CSJ_91stations/processed_meteroloy_data_ws+wd/m20240' + str(
#                     i) + '.nc')
#
#             if i == 1:
#                 longitude = np.array((nc_obj.variables['longitude'][:]))
#                 latitude = np.array((nc_obj.variables['latitude'][:]))
#                 time = np.concatenate((time_2023, np.array((nc_obj.variables['valid_time'][:]))), axis=0)
#                 t2m = np.concatenate((t2m_2023, np.array((nc_obj.variables['t2m'][:]))), axis=0)
#                 ws = np.concatenate((ws_2023, np.array((nc_obj.variables['ws'][:]))), axis=0)
#                 wd = np.concatenate((wd_2023, np.array((nc_obj.variables['wd'][:]))), axis=0)
#                 d2m = np.concatenate((d2m_2023, np.array((nc_obj.variables['d2m'][:]))), axis=0)
#                 sp = np.concatenate((sp_2023, np.array((nc_obj.variables['sp'][:]))), axis=0)
#                 tp = np.concatenate((tp_2023, np.array((nc_obj.variables['tp'][:]))), axis=0)
#             else:
#                 time = np.concatenate((time, np.array((nc_obj.variables['valid_time'][:]))), axis=0)
#                 t2m = np.concatenate((t2m, np.array((nc_obj.variables['t2m'][:]))), axis=0)
#                 ws = np.concatenate((ws, np.array((nc_obj.variables['ws'][:]))), axis=0)
#                 wd = np.concatenate((wd, np.array((nc_obj.variables['wd'][:]))), axis=0)
#                 d2m = np.concatenate((d2m, np.array((nc_obj.variables['d2m'][:]))), axis=0)
#                 sp = np.concatenate((sp, np.array((nc_obj.variables['sp'][:]))), axis=0)
#                 tp = np.concatenate((tp, np.array((nc_obj.variables['tp'][:]))), axis=0)
#         print(time.shape)
#         print(longitude.shape)
#         print(latitude.shape)
#
#
#         dAQI_2023 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_AQI_91stations_2023.csv')
#         dCO_2023 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_CO_91stations_2023.csv')
#         dNO2_2023 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_NO2_91stations_2023.csv')
#         dO3_2023 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_O3_91stations_2023.csv')
#         dPM10_2023 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM10_91stations_2023.csv')
#         dPM25_2023 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM2.5_91stations_2023.csv')
#         dSO2_2023 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_SO2_91stations_2023.csv')
#
#
#         dAQI = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_AQI_91stations_2024.csv')
#         dCO = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_CO_91stations_2024.csv')
#         dNO2 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_NO2_91stations_2024.csv')
#         dO3 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_O3_91stations_2024.csv')
#         dPM10 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM10_91stations_2024.csv')
#         dPM25 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM2.5_91stations_2024.csv')
#         dSO2 = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_SO2_91stations_2024.csv')
#
#
#         dstation = pd.read_csv(
#             '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_91stationcoordinates.csv')
#         coordinates = dstation[['latitude', 'longitude']]
#         coordinates = np.array(coordinates).reshape(-1, 2)
#         print('coordinates.shape:', coordinates.shape)
#
#         dt2m = np.zeros((t2m.shape[0], coordinates.shape[0]))
#         dws = np.zeros((ws.shape[0], coordinates.shape[0]))
#         dwd = np.zeros((wd.shape[0], coordinates.shape[0]))
#         dd2m = np.zeros((d2m.shape[0], coordinates.shape[0]))
#         dsp = np.zeros((sp.shape[0], coordinates.shape[0]))
#         dtp = np.zeros((tp.shape[0], coordinates.shape[0]))
#
#         wlongitude = []
#         wlatitude = []
#
#         for i in range(coordinates.shape[0]):
#             wherelongitude = np.argmin(np.abs(longitude - coordinates[i, 1]))
#             wherelatitude = np.argmin(np.abs(latitude - coordinates[i, 0]))
#             wlongitude.append(wherelongitude)
#             wlatitude.append(wherelatitude)
#         print('wlatitude:', wlatitude)
#         print('wlongitude:', wlongitude)
#         wlongitude = np.array(wlongitude).reshape(coordinates.shape[0])
#         wlatitude = np.array(wlatitude).reshape(coordinates.shape[0])
#
#
#         for k in range(t2m.shape[0]):
#             for j in range(coordinates.shape[0]):
#                 dt2m[k][j] = t2m[k][wlatitude[j]][wlongitude[j]]
#                 dws[k][j] = ws[k][wlatitude[j]][wlongitude[j]]
#                 dwd[k][j] = wd[k][wlatitude[j]][wlongitude[j]]
#                 dd2m[k][j] = d2m[k][wlatitude[j]][wlongitude[j]]
#                 dsp[k][j] = sp[k][wlatitude[j]][wlongitude[j]]
#                 dtp[k][j] = tp[k][wlatitude[j]][wlongitude[j]]
#
#         dAQI = dAQI[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                      '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                      '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                      '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                      '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                      '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                      '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                      '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                      '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                      '1279A']].interpolate(method='linear', limit_direction='forward')
#         dCO = dCO[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                    '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                    '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                    '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                    '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                    '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                    '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                    '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                    '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                    '1279A']].interpolate(method='linear', limit_direction='forward')
#         dNO2 = dNO2[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                      '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                      '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                      '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                      '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                      '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                      '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                      '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                      '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                      '1279A']].interpolate(method='linear', limit_direction='forward')
#         dO3 = dO3[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                    '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                    '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                    '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                    '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                    '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                    '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                    '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                    '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                    '1279A']].interpolate(method='linear', limit_direction='forward')
#         dPM10 = dPM10[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                        '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                        '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                        '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                        '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                        '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                        '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                        '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                        '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                        '1279A']].interpolate(method='linear', limit_direction='forward')
#         dPM25 = dPM25[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                        '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                        '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                        '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                        '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                        '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                        '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                        '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                        '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                        '1279A']].interpolate(method='linear', limit_direction='forward')
#         dSO2 = dSO2[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                      '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                      '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                      '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                      '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                      '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                      '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                      '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                      '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                      '1279A']].interpolate(method='linear', limit_direction='forward')
#
#         dAQI_2023 = dAQI_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                                '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                                '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                                '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                                '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                                '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                                '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                                '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                                '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                                '1279A']].interpolate(method='linear', limit_direction='forward')
#         dCO_2023 = dCO_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                              '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                              '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                              '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                              '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                              '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                              '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                              '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                              '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                              '1279A']].interpolate(method='linear', limit_direction='forward')
#         dNO2_2023 = dNO2_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                                '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                                '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                                '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                                '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                                '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                                '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                                '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                                '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                                '1279A']].interpolate(method='linear', limit_direction='forward')
#         dO3_2023 = dO3_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                              '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                              '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                              '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                              '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                              '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                              '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                              '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                              '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                              '1279A']].interpolate(method='linear', limit_direction='forward')
#         dPM10_2023 = dPM10_2023[
#             ['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#              '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#              '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#              '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#              '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#              '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#              '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#              '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#              '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#              '1279A']].interpolate(method='linear', limit_direction='forward')
#         dPM25_2023 = dPM25_2023[
#             ['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#              '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#              '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#              '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#              '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#              '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#              '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#              '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#              '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#              '1279A']].interpolate(method='linear', limit_direction='forward')
#         dSO2_2023 = dSO2_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
#                                '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
#                                '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
#                                '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
#                                '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
#                                '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
#                                '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
#                                '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
#                                '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
#                                '1279A']].interpolate(method='linear', limit_direction='forward')
#
#         dAQI = dAQI.interpolate(method='linear', limit_direction='backward')
#         dCO = dCO.interpolate(method='linear', limit_direction='backward')
#         dNO2 = dNO2.interpolate(method='linear', limit_direction='backward')
#         dO3 = dO3.interpolate(method='linear', limit_direction='backward')
#         dPM10 = dPM10.interpolate(method='linear', limit_direction='backward')
#         dPM25 = dPM25.interpolate(method='linear', limit_direction='backward')
#         dSO2 = dSO2.interpolate(method='linear', limit_direction='backward')
#         dAQI_2023 = dAQI_2023.interpolate(method='linear', limit_direction='backward')
#         dCO_2023 = dCO_2023.interpolate(method='linear', limit_direction='backward')
#         dNO2_2023 = dNO2_2023.interpolate(method='linear', limit_direction='backward')
#         dO3_2023 = dO3_2023.interpolate(method='linear', limit_direction='backward')
#         dPM10_2023 = dPM10_2023.interpolate(method='linear', limit_direction='backward')
#         dPM25_2023 = dPM25_2023.interpolate(method='linear', limit_direction='backward')
#         dSO2_2023 = dSO2_2023.interpolate(method='linear', limit_direction='backward')
#
#
#         daqi = np.array(dAQI)
#         dco = np.array(dCO)
#         dno2 = np.array(dNO2)
#         do3 = np.array(dO3)
#         dpm10 = np.array(dPM10)
#         dpm25 = np.array(dPM25)
#         dso2 = np.array(dSO2)
#
#         daqi_2023 = np.array(dAQI_2023)
#         dco_2023 = np.array(dCO_2023)
#         dno2_2023 = np.array(dNO2_2023)
#         do3_2023 = np.array(dO3_2023)
#         dpm10_2023 = np.array(dPM10_2023)
#         dpm25_2023 = np.array(dPM25_2023)
#         dso2_2023 = np.array(dSO2_2023)
#
#
#         daqi = np.concatenate((daqi_2023[:, :], daqi), 0)
#         dco = np.concatenate((dco_2023[:, :], dco), 0)
#         dno2 = np.concatenate((dno2_2023[:, :], dno2), 0)
#         do3 = np.concatenate((do3_2023[:, :], do3), 0)
#         dpm10 = np.concatenate((dpm10_2023[:, :], dpm10), 0)
#         dpm25 = np.concatenate((dpm25_2023[:, :], dpm25), 0)
#         dso2 = np.concatenate((dso2_2023[:, :], dso2), 0)
#         do3 = np.nan_to_num(do3)
#         df = pd.DataFrame(dpm25)
#         df.to_csv('/tmp/pycharm_project_613/dataset/tmp/pycharm_project_613/data_PM25.csv', )
#
#
#         num_features = 13
#         data = np.zeros((daqi.shape[0], 1))
#
#         for i in range(coordinates.shape[0]):
#             data = np.concatenate((data,
#                                    dt2m[:, i].reshape(-1, 1),
#                                    dws[:, i].reshape(-1, 1), dwd[:, i].reshape(-1, 1),
#                                    dd2m[:, i].reshape(-1, 1),
#                                    dsp[:, i].reshape(-1, 1),
#                                    dtp[:, i].reshape(-1, 1),
#                                    daqi[:, i].reshape(-1, 1),
#                                    dco[:, i].reshape(-1, 1),
#                                    dno2[:, i].reshape(-1, 1),
#                                    do3[:, i].reshape(-1, 1),
#                                    dpm10[:, i].reshape(-1, 1),
#                                    dso2[:, i].reshape(-1, 1),
#                                    dpm25[:, i].reshape(-1, 1)), axis=1)
#
#         data = np.delete(data, obj=0, axis=1)
#         data_final = np.zeros((coordinates.shape[0], daqi.shape[0], num_features))
#         for i in range(coordinates.shape[0]):
#             data_final[i] = data[:, i * num_features:(i + 1) * num_features]
#
#         X = data_final
#         data_normal = []
#         num_nodes = data_final.shape[0]
#         features = data_final.shape[2]
#         x_max, y_max, x_min, y_min = 0, 0, 0, 0
#         Y = data_final[:, :, -1].reshape(-1, 1)
#         for i in range(data_final.shape[0]):
#             data_temp, x_max, x_min = self._Maxmin(data_final[i, :, :])
#             data_normal.append(data_temp)
#         data = np.concatenate((data_normal), axis=0)
#
#
#         Y = Y.reshape(num_nodes, -1).T
#         data = data.reshape(num_nodes, -1, features)
#         X = data.transpose(1, 0, 2)
#         data_X = X[0:100, 38:41, :7]
#
#         return X, Y, X.shape[0], X.shape[1], X.shape[2], coordinates, longitude, latitude
#
#
#     def _getAdj(self):
#         c = []
#         for i in range(self.coordinates.shape[0]):
#             c.append((self.coordinates[i, 0], self.coordinates[i, 1]))
#         station_distance = haversine_vector(c, c, Unit.KILOMETERS, comb=True).reshape(self.coordinates.shape[0],
#                                                                                       -1)
#         adj = np.zeros_like(station_distance)
#         for i in range(self.coordinates.shape[0]):
#             for j in range(self.coordinates.shape[0]):
#                 if i == j:
#                     adj[i, j] = 0
#                 elif station_distance[i, j] < 222.24:
#                     adj[i, j] = np.around(1/station_distance[i, j], decimals=4)
#                 else:
#                     adj[i, j] = 0
#         np.fill_diagonal(adj, 1.0)
#         df = pd.DataFrame(adj)
#         df.to_csv('graph_matrix.csv')
#         adj = torch.from_numpy(adj).float()
#         if torch.cuda.is_available():
#             adj = adj.cuda()
#         return adj
#
#     def _Maxmin(self, data):
#         maxdata = np.max(data, axis=0)
#         mindata = np.min(data, axis=0)
#         denominator = maxdata - mindata
#         for i in range(denominator.shape[0]):
#             if denominator[i] == 0:
#                 denominator[i] += 1e-8
#         return (data - mindata) / denominator, maxdata, mindata
#
#     def _split(self, train, valid):
#         print('trian:', train, 'valid:', valid)  # trian: 10512 valid: 14016
#         train_set = range(self.input_seq_len + self.output_seq_len - 1, train)
#         valid_set = range(train, valid)
#         test_set = range(valid, self.num)  # 最后一部分数据是测试集
#         print('train_set:', train_set, 'valid_set:', valid_set, 'test_set:', test_set)
#         print('len_train_set:', len(train_set), 'len_valide:', len(valid_set), 'len_test:', len(test_set))
#         self.train = self._batchify(train_set, self.output_seq_len)
#         self.valid = self._batchify(valid_set, self.output_seq_len)
#         self.test = self._batchify(test_set, self.output_seq_len)
#
#     def _batchify(self, idx_set, horizon):
#         n = len(idx_set)
#         dataX = torch.zeros((n, self.input_seq_len, self.num_nodes, self.features))
#         dataY = torch.zeros((n, self.output_seq_len, self.num_nodes))
#         for i in range(n):
#             end = idx_set[i] - self.output_seq_len + 1  #
#             start = end - self.input_seq_len
#             dataX[i, :, :, :] = torch.from_numpy(self.X[start:end, :, :])
#             if self.output_seq_len == 1:
#                 dataY[i, :, :] = torch.from_numpy(self.Y[end, :])
#             else:
#                 dataY[i, :, :] = torch.from_numpy(self.Y[end:end + self.output_seq_len, :])
#
#         if torch.cuda.is_available():
#             dataX = dataX.cuda()
#             dataY = dataY.cuda()
#         return [dataX, dataY]
#
#     def get_batches(self, inputs, targets, batch_size, shuffle=True):
#         length = len(inputs)
#         if shuffle:
#             index = torch.randperm(length)
#         else:
#             index = torch.LongTensor(range(length))
#         start_idx = 0
#         while (start_idx < length):
#             end_idx = min(length, start_idx + batch_size)
#             excerpt = index[start_idx:end_idx]
#             X = inputs[excerpt]
#             Y = targets[excerpt]
#             yield Variable(X), Variable(Y)
#             start_idx += batch_size
#
#
# if __name__ == "__main__":
#     train_ratio, valid_ratio, input_seq_len, output_seq_len = 0.6, 0.2, 48, 1
#     Data = LoadData(train_ratio, valid_ratio, input_seq_len, output_seq_len)
#     print(Data.X.shape, Data.Y.shape)
import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch
from haversine import haversine_vector, Unit
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS ']


class LoadData():
    def __init__(self, train_ratio, valid_ratio, input_seq_len, output_seq_len):
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        self.X, self.Y, self.num, self.num_nodes, self.features, self.coordinates, self.mete_long, self.mete_lat = self._load()

        train_end = int(self.train_ratio * self.num)
        self.X = self._global_zscore(self.X, train_end)

        self._split(int(self.train_ratio * self.num), int((self.valid_ratio + self.train_ratio) * self.num))
        self.adj = self._getAdj()

    def get_azimuth(self):
        coords = self.coordinates  # numpy: (N,2) -> [lat, lon]
        N = self.num_nodes
        az = torch.zeros((N, N), dtype=torch.long)

        code_map = [5, 3, 8, 2, 6, 4, 7, 1]

        for i in range(N):
            lat_i, lon_i = coords[i, 0], coords[i, 1]
            for j in range(N):
                if i == j:
                    continue
                lat_j, lon_j = coords[j, 0], coords[j, 1]
                dy = lat_j - lat_i
                dx = lon_j - lon_i
                if dx == 0 and dy == 0:
                    continue
                angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                sector = int(((angle + 22.5) // 45) % 8)  # 0..7
                az[i, j] = code_map[sector]

        return az

    def dimensionlessProcessing(self, df_values):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        res = scaler.fit_transform(df_values)
        return pd.DataFrame(res)

    def _load(self):
        for i in range(1, 13):

            nc_obj_2023 = Dataset(
                '/tmp/pycharm_project_613/dataset/CSJ_91stations/processed_meteroloy_data_ws+wd/m20230' + str(
                    i) + '.nc')
            if i == 1:
                longitude_2023 = np.array((nc_obj_2023.variables['longitude'][:]))
                latitude_2023 = np.array((nc_obj_2023.variables['latitude'][:]))
                time_2023 = np.array((nc_obj_2023.variables['valid_time'][:]))
                t2m_2023 = np.array((nc_obj_2023.variables['t2m'][:]))
                ws_2023 = np.array((nc_obj_2023.variables['ws'][:]))
                wd_2023 = np.array((nc_obj_2023.variables['wd'][:]))
                d2m_2023 = np.array((nc_obj_2023.variables['d2m'][:]))
                sp_2023 = np.array((nc_obj_2023.variables['sp'][:]))
                tp_2023 = np.array((nc_obj_2023.variables['tp'][:]))
            else:
                time_2023 = np.concatenate((time_2023, np.array((nc_obj_2023.variables['valid_time'][:]))), axis=0)
                t2m_2023 = np.concatenate((t2m_2023, np.array((nc_obj_2023.variables['t2m'][:]))), axis=0)
                ws_2023 = np.concatenate((ws_2023, np.array((nc_obj_2023.variables['ws'][:]))), axis=0)
                wd_2023 = np.concatenate((wd_2023, np.array((nc_obj_2023.variables['wd'][:]))), axis=0)
                d2m_2023 = np.concatenate((d2m_2023, np.array((nc_obj_2023.variables['d2m'][:]))), axis=0)
                sp_2023 = np.concatenate((sp_2023, np.array((nc_obj_2023.variables['sp'][:]))), axis=0)
                tp_2023 = np.concatenate((tp_2023, np.array((nc_obj_2023.variables['tp'][:]))), axis=0)

        for i in range(1, 13):
            nc_obj = Dataset(
                '/tmp/pycharm_project_613/dataset/CSJ_91stations/processed_meteroloy_data_ws+wd/m20240' + str(
                    i) + '.nc')

            if i == 1:
                longitude = np.array((nc_obj.variables['longitude'][:]))
                latitude = np.array((nc_obj.variables['latitude'][:]))
                time = np.concatenate((time_2023, np.array((nc_obj.variables['valid_time'][:]))), axis=0)
                t2m = np.concatenate((t2m_2023, np.array((nc_obj.variables['t2m'][:]))), axis=0)
                ws = np.concatenate((ws_2023, np.array((nc_obj.variables['ws'][:]))), axis=0)
                wd = np.concatenate((wd_2023, np.array((nc_obj.variables['wd'][:]))), axis=0)
                d2m = np.concatenate((d2m_2023, np.array((nc_obj.variables['d2m'][:]))), axis=0)
                sp = np.concatenate((sp_2023, np.array((nc_obj.variables['sp'][:]))), axis=0)
                tp = np.concatenate((tp_2023, np.array((nc_obj.variables['tp'][:]))), axis=0)
            else:
                time = np.concatenate((time, np.array((nc_obj.variables['valid_time'][:]))), axis=0)
                t2m = np.concatenate((t2m, np.array((nc_obj.variables['t2m'][:]))), axis=0)
                ws = np.concatenate((ws, np.array((nc_obj.variables['ws'][:]))), axis=0)
                wd = np.concatenate((wd, np.array((nc_obj.variables['wd'][:]))), axis=0)
                d2m = np.concatenate((d2m, np.array((nc_obj.variables['d2m'][:]))), axis=0)
                sp = np.concatenate((sp, np.array((nc_obj.variables['sp'][:]))), axis=0)
                tp = np.concatenate((tp, np.array((nc_obj.variables['tp'][:]))), axis=0)
        print(time.shape)
        print(longitude.shape)
        print(latitude.shape)


        dAQI_2023 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_AQI_91stations_2023.csv')
        dCO_2023 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_CO_91stations_2023.csv')
        dNO2_2023 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_NO2_91stations_2023.csv')
        dO3_2023 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_O3_91stations_2023.csv')
        dPM10_2023 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM10_91stations_2023.csv')
        dPM25_2023 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM2.5_91stations_2023.csv')
        dSO2_2023 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_SO2_91stations_2023.csv')


        dAQI = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_AQI_91stations_2024.csv')
        dCO = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_CO_91stations_2024.csv')
        dNO2 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_NO2_91stations_2024.csv')
        dO3 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_O3_91stations_2024.csv')
        dPM10 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM10_91stations_2024.csv')
        dPM25 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_PM2.5_91stations_2024.csv')
        dSO2 = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_pollute_data/CSJ_SO2_91stations_2024.csv')


        dstation = pd.read_csv(
            '/tmp/pycharm_project_613/dataset/CSJ_91stations/CSJ_91stationcoordinates.csv')
        coordinates = dstation[['latitude', 'longitude']]
        coordinates = np.array(coordinates).reshape(-1, 2)
        print('coordinates.shape:', coordinates.shape)

        dt2m = np.zeros((t2m.shape[0], coordinates.shape[0]))
        dws = np.zeros((ws.shape[0], coordinates.shape[0]))
        dwd = np.zeros((wd.shape[0], coordinates.shape[0]))
        dd2m = np.zeros((d2m.shape[0], coordinates.shape[0]))
        dsp = np.zeros((sp.shape[0], coordinates.shape[0]))
        dtp = np.zeros((tp.shape[0], coordinates.shape[0]))

        wlongitude = []
        wlatitude = []

        for i in range(coordinates.shape[0]):
            wherelongitude = np.argmin(np.abs(longitude - coordinates[i, 1]))
            wherelatitude = np.argmin(np.abs(latitude - coordinates[i, 0]))
            wlongitude.append(wherelongitude)
            wlatitude.append(wherelatitude)
        print('wlatitude:', wlatitude)
        print('wlongitude:', wlongitude)
        wlongitude = np.array(wlongitude).reshape(coordinates.shape[0])
        wlatitude = np.array(wlatitude).reshape(coordinates.shape[0])


        for k in range(t2m.shape[0]):
            for j in range(coordinates.shape[0]):
                dt2m[k][j] = t2m[k][wlatitude[j]][wlongitude[j]]
                dws[k][j] = ws[k][wlatitude[j]][wlongitude[j]]
                dwd[k][j] = wd[k][wlatitude[j]][wlongitude[j]]
                dd2m[k][j] = d2m[k][wlatitude[j]][wlongitude[j]]
                dsp[k][j] = sp[k][wlatitude[j]][wlongitude[j]]
                dtp[k][j] = tp[k][wlatitude[j]][wlongitude[j]]

        dAQI = dAQI[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                     '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                     '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                     '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                     '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                     '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                     '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                     '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                     '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                     '1279A']].interpolate(method='linear', limit_direction='forward')
        dCO = dCO[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                   '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                   '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                   '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                   '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                   '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                   '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                   '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                   '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                   '1279A']].interpolate(method='linear', limit_direction='forward')
        dNO2 = dNO2[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                     '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                     '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                     '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                     '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                     '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                     '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                     '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                     '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                     '1279A']].interpolate(method='linear', limit_direction='forward')
        dO3 = dO3[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                   '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                   '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                   '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                   '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                   '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                   '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                   '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                   '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                   '1279A']].interpolate(method='linear', limit_direction='forward')
        dPM10 = dPM10[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                       '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                       '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                       '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                       '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                       '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                       '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                       '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                       '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                       '1279A']].interpolate(method='linear', limit_direction='forward')
        dPM25 = dPM25[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                       '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                       '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                       '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                       '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                       '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                       '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                       '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                       '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                       '1279A']].interpolate(method='linear', limit_direction='forward')
        dSO2 = dSO2[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                     '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                     '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                     '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                     '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                     '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                     '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                     '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                     '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                     '1279A']].interpolate(method='linear', limit_direction='forward')

        dAQI_2023 = dAQI_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                               '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                               '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                               '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                               '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                               '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                               '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                               '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                               '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                               '1279A']].interpolate(method='linear', limit_direction='forward')
        dCO_2023 = dCO_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                             '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                             '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                             '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                             '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                             '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                             '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                             '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                             '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                             '1279A']].interpolate(method='linear', limit_direction='forward')
        dNO2_2023 = dNO2_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                               '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                               '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                               '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                               '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                               '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                               '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                               '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                               '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                               '1279A']].interpolate(method='linear', limit_direction='forward')
        dO3_2023 = dO3_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                             '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                             '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                             '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                             '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                             '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                             '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                             '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                             '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                             '1279A']].interpolate(method='linear', limit_direction='forward')
        dPM10_2023 = dPM10_2023[
            ['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
             '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
             '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
             '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
             '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
             '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
             '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
             '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
             '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
             '1279A']].interpolate(method='linear', limit_direction='forward')
        dPM25_2023 = dPM25_2023[
            ['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
             '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
             '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
             '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
             '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
             '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
             '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
             '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
             '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
             '1279A']].interpolate(method='linear', limit_direction='forward')
        dSO2_2023 = dSO2_2023[['1141A', '1142A', '1143A', '1144A', '1145A', '1147A', '1148A', '1149A', '1150A', '1151A',
                               '1152A', '1153A', '1154A', '1155A', '1156A', '1157A', '1158A', '1159A', '1160A', '1162A',
                               '1163A', '1164A', '1165A', '1166A', '1167A', '1168A', '1169A', '1170A', '1171A', '1172A',
                               '1186A', '1188A', '1189A', '1190A', '1191A', '1192A', '1193A', '1194A', '1195A', '1196A',
                               '1200A', '1201A', '1203A', '1204A', '1205A', '1206A', '1207A', '1209A', '1215A', '1216A',
                               '1218A', '1223A', '1224A', '1226A', '1227A', '1228A', '1230A', '1231A', '1232A', '1233A',
                               '1234A', '1235A', '1236A', '1239A', '1240A', '1241A', '1242A', '1243A', '1244A', '1245A',
                               '1247A', '1250A', '1252A', '1253A', '1255A', '1256A', '1257A', '1258A', '1259A', '1260A',
                               '1262A', '1263A', '1270A', '1271A', '1272A', '1274A', '1275A', '1276A', '1277A', '1278A',
                               '1279A']].interpolate(method='linear', limit_direction='forward')

        dAQI = dAQI.interpolate(method='linear', limit_direction='backward')
        dCO = dCO.interpolate(method='linear', limit_direction='backward')
        dNO2 = dNO2.interpolate(method='linear', limit_direction='backward')
        dO3 = dO3.interpolate(method='linear', limit_direction='backward')
        dPM10 = dPM10.interpolate(method='linear', limit_direction='backward')
        dPM25 = dPM25.interpolate(method='linear', limit_direction='backward')
        dSO2 = dSO2.interpolate(method='linear', limit_direction='backward')
        dAQI_2023 = dAQI_2023.interpolate(method='linear', limit_direction='backward')
        dCO_2023 = dCO_2023.interpolate(method='linear', limit_direction='backward')
        dNO2_2023 = dNO2_2023.interpolate(method='linear', limit_direction='backward')
        dO3_2023 = dO3_2023.interpolate(method='linear', limit_direction='backward')
        dPM10_2023 = dPM10_2023.interpolate(method='linear', limit_direction='backward')
        dPM25_2023 = dPM25_2023.interpolate(method='linear', limit_direction='backward')
        dSO2_2023 = dSO2_2023.interpolate(method='linear', limit_direction='backward')


        daqi = np.array(dAQI)
        dco = np.array(dCO)
        dno2 = np.array(dNO2)
        do3 = np.array(dO3)
        dpm10 = np.array(dPM10)
        dpm25 = np.array(dPM25)
        dso2 = np.array(dSO2)

        daqi_2023 = np.array(dAQI_2023)
        dco_2023 = np.array(dCO_2023)
        dno2_2023 = np.array(dNO2_2023)
        do3_2023 = np.array(dO3_2023)
        dpm10_2023 = np.array(dPM10_2023)
        dpm25_2023 = np.array(dPM25_2023)
        dso2_2023 = np.array(dSO2_2023)


        daqi = np.concatenate((daqi_2023[:, :], daqi), 0)
        dco = np.concatenate((dco_2023[:, :], dco), 0)
        dno2 = np.concatenate((dno2_2023[:, :], dno2), 0)
        do3 = np.concatenate((do3_2023[:, :], do3), 0)
        dpm10 = np.concatenate((dpm10_2023[:, :], dpm10), 0)
        dpm25 = np.concatenate((dpm25_2023[:, :], dpm25), 0)
        dso2 = np.concatenate((dso2_2023[:, :], dso2), 0)
        do3 = np.nan_to_num(do3)
        df = pd.DataFrame(dpm25)
        df.to_csv('/tmp/pycharm_project_613/data_PM25.csv', )


        num_features = 13
        data = np.zeros((daqi.shape[0], 1))

        for i in range(coordinates.shape[0]):
            data = np.concatenate((data,
                                   dt2m[:, i].reshape(-1, 1),
                                   dws[:, i].reshape(-1, 1), dwd[:, i].reshape(-1, 1),
                                   dd2m[:, i].reshape(-1, 1),
                                   dsp[:, i].reshape(-1, 1),
                                   dtp[:, i].reshape(-1, 1),
                                   daqi[:, i].reshape(-1, 1),
                                   dco[:, i].reshape(-1, 1),
                                   dno2[:, i].reshape(-1, 1),
                                   do3[:, i].reshape(-1, 1),
                                   dpm10[:, i].reshape(-1, 1),
                                   dso2[:, i].reshape(-1, 1),
                                   dpm25[:, i].reshape(-1, 1)), axis=1)

        data = np.delete(data, obj=0, axis=1)
        data_final = np.zeros((coordinates.shape[0], daqi.shape[0], num_features))
        for i in range(coordinates.shape[0]):
            data_final[i] = data[:, i * num_features:(i + 1) * num_features]

        num_nodes = data_final.shape[0]
        features = data_final.shape[2]

        X = data_final.transpose(1, 0, 2)
        Y = data_final[:, :, -1].T

        return X, Y, X.shape[0], X.shape[1], X.shape[2], coordinates, longitude, latitude

    def _global_zscore(self, X, train_end):
        train_X = X[:train_end, :, :]
        mean = train_X.reshape(-1, X.shape[-1]).mean(axis=0)
        std = train_X.reshape(-1, X.shape[-1]).std(axis=0)
        std[std == 0] = 1e-8
        return (X - mean) / std

    def _getAdj(self):
        c = []
        for i in range(self.coordinates.shape[0]):
            c.append((self.coordinates[i, 0], self.coordinates[i, 1]))
        station_distance = haversine_vector(c, c, Unit.KILOMETERS, comb=True).reshape(self.coordinates.shape[0],
                                                                                      -1)
        adj = np.zeros_like(station_distance)
        for i in range(self.coordinates.shape[0]):
            for j in range(self.coordinates.shape[0]):
                if i == j:
                    adj[i, j] = 0
                elif station_distance[i, j] < 222.24:
                    adj[i, j] = np.around(1/station_distance[i, j], decimals=4)
                else:
                    adj[i, j] = 0
        np.fill_diagonal(adj, 1.0)
        df = pd.DataFrame(adj)
        df.to_csv('graph_matrix.csv')
        adj = torch.from_numpy(adj).float()
        if torch.cuda.is_available():
            adj = adj.cuda()
        return adj

    def _Maxmin(self, data):
        maxdata = np.max(data, axis=0)
        mindata = np.min(data, axis=0)
        denominator = maxdata - mindata
        for i in range(denominator.shape[0]):
            if denominator[i] == 0:
                denominator[i] += 1e-8
        return (data - mindata) / denominator, maxdata, mindata

    def _split(self, train, valid):
        print('trian:', train, 'valid:', valid)  # trian: 10512 valid: 14016
        train_set = range(self.input_seq_len + self.output_seq_len - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.num)  # 最后一部分数据是测试集
        print('train_set:', train_set, 'valid_set:', valid_set, 'test_set:', test_set)
        print('len_train_set:', len(train_set), 'len_valide:', len(valid_set), 'len_test:', len(test_set))
        self.train = self._batchify(train_set, self.output_seq_len)
        self.valid = self._batchify(valid_set, self.output_seq_len)
        self.test = self._batchify(test_set, self.output_seq_len)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        dataX = torch.zeros((n, self.input_seq_len, self.num_nodes, self.features))
        dataY = torch.zeros((n, self.output_seq_len, self.num_nodes))
        for i in range(n):
            end = idx_set[i] - self.output_seq_len + 1  #
            start = end - self.input_seq_len
            dataX[i, :, :, :] = torch.from_numpy(self.X[start:end, :, :])
            if self.output_seq_len == 1:
                dataY[i, :, :] = torch.from_numpy(self.Y[end, :])
            else:
                dataY[i, :, :] = torch.from_numpy(self.Y[end:end + self.output_seq_len, :])

        if torch.cuda.is_available():
            dataX = dataX.cuda()
            dataY = dataY.cuda()
        return [dataX, dataY]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            yield Variable(X), Variable(Y)
            start_idx += batch_size


if __name__ == "__main__":
    train_ratio, valid_ratio, input_seq_len, output_seq_len = 0.6, 0.2, 48, 1
    Data = LoadData(train_ratio, valid_ratio, input_seq_len, output_seq_len)
    print(Data.X.shape, Data.Y.shape)