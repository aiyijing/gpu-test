FROM tensorflow/tensorflow:latest-gpu

ADD test.py /tf/
ADD start.sh /tf/

WORKDIR /tf/
RUN chmod a+x start.sh && chmod a+x test.py


CMD [ "/tf/start.sh" ]
