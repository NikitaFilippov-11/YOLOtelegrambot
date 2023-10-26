FROM ultralytics/ultralytics:latest-cpu

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg

# Aplication root directory
RUN mkdir /usr/share/app
RUN mkdir /usr/share/app/photos
RUN mkdir /usr/share/app/videos
RUN mkdir /usr/share/app/videos/model_videos
RUN mkdir /usr/share/app/videos/send
# Aplication logs root directory
RUN mkdir /usr/share/app/logs
# Mount logs directory to volume
VOLUME /usr/share/app/logs
# Set application root as a work directory
WORKDIR /usr/share/app/
# Copy artifact <app-name>-<app-version>.jar to /app/root/directory/app.jar
COPY main.py /usr/share/app/main.py
COPY requirements.txt /usr/share/app/requirements.txt
COPY yolov5s.pt /usr/share/app/yolov5s.pt

RUN pip install -r requirements.txt

CMD [ "python3", "main.py"]