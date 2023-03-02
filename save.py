#self.cap = cv2.VideoCapture(os.path.dirname(__file__) + "/video/input/videoTribord.mp4")
#self.cap2 = cv2.VideoCapture(os.path.dirname(__file__) + "/video/input/videoBabord.mp4")
self.cap = cv2.VideoCapture("rtsp://root:M101_svr@192.168.1.56:554/axis-media/media.amp") # 192.168.1.55 ou 192.168.1.56
self.cap2 = cv2.VideoCapture("rtsp://root:M101_svr@192.168.1.56:554/axis-media/media.amp") # 192.168.1.55 ou 192.168.1.56

# Liste des zones d'analyses
self.analyses = {}
self.analyses["derive"] = AnalyseMousse(1462, 271, 1514, 410, 27)
self.analyses["safran"] = AnalyseSafran(355, 433, 451, 572, 411, 435, 423, 506, 21)
self.analyses["mousse"] = AnalyseMousse(570, 380, 715, 637, 31)

self.analysesTribord = {}
self.analysesTribord["derive"] = AnalyseMousse(878, 423, 895, 484, 20)
self.analysesTribord["safran"] = AnalyseSafran(520, 406, 609, 549, 562, 413, 578, 467, 12)
self.analysesTribord["mousse"] = AnalyseMousse(632, 415, 680, 526, 14)

self.analysesBabord = {}
self.analysesBabord["derive"] = AnalyseMousse(850, 568, 888, 654, 25)
self.analysesBabord["safran"] = AnalyseSafran(1234, 537, 1337, 669, 1315, 550, 1262, 645, 23)
self.analysesBabord["mousse"] = AnalyseMousse(1170, 555, 1218, 665, 25)

# liste des threads
self.analyseVideoList = []
self.analyseVideoList.append(AnalyseVideo(self.cap, self.analysesTribord, "Tribord"))
self.analyseVideoList.append(AnalyseVideo(self.cap2, self.analysesBabord, "Babord"))

self.listData = {}