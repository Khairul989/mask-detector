import cv2
import uuid
import json
from ibm_watson import VisualRecognitionV4
from ibm_watson.visual_recognition_v4 import FileWithMetadata, AnalyzeEnums
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# while True:
#     ret, frame = cap.read()
#     imgname = './Images/Nomask/{}.jpg'.format(str(uuid.uuid1()))
#     cv2.imwrite(imgname, frame)
#     cv2.imshow('frame',frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cap.destroyAllWindows()

#Scoring
apikey = 'r698aMTiJ_wRRiDzTq48nFXqn1pQrXBIlyYb-N_mhzO0'
url = 'https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/6bdd40fd-b851-4d76-b9f9-919fe01b38d2'
collection = 'b22686c0-e62b-4d74-911d-629f798d40dc'

authenticator = IAMAuthenticator(apikey)
service = VisualRecognitionV4('2021-01-11', authenticator=authenticator)
service.set_service_url(url)

path = './Images/Mask/010c0906-5366-11eb-afb9-d639f59b7504.jpg'

with open(path, 'rb') as mask_img:
    analyze_img = service.analyze(collection_ids=[collection], features=[AnalyzeEnums.Features.OBJECTS.value], images_file=[FileWithMetadata(mask_img)]).get_result()

analyze_img