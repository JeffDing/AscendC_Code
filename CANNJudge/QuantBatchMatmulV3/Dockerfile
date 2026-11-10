FROM huawei0809-registry-huabei2.crs.ctyun.cn/openmind-space-base/openmind-cann:openeuler-python3.12-cann9.1.0.beta3

USER root

# 统一 pip 版本，解决 "--root-user-action" 不兼容问题
# 在安装任何包之前，先将 pip 自身升级到最新版本。
RUN mkdir -p ~/.pip && \
    printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n" > ~/.pip/pip.conf
RUN pip install --upgrade --ignore-installed pip

# no packages to install


RUN set -e; \
    if ! getent group openmind > /dev/null; then \
        groupadd -g 1000 openmind; \
    fi; \
    if ! id -u openmind > /dev/null; then \
        useradd -u 1000 -g openmind -d /home/openmind -m -s /bin/bash openmind; \
    fi


ENV WORKDIR=/home/openmind
ENV CODEDIR=/home/openmind/code

WORKDIR ${WORKDIR}

COPY --chown=openmind . ${CODEDIR}

RUN pip install --root-user-action=ignore -r ${CODEDIR}/requirements.txt

RUN pip show jupyter-server | grep 'Location:' | awk '{print $2}' > ${WORKDIR}/location.txt

RUN export JUPYTER_SERVER_LOCATION=$(cat ${WORKDIR}/location.txt)/jupyter_server/templates && \
    cp -f ${CODEDIR}/login.html $JUPYTER_SERVER_LOCATION/login.html

RUN rm ${WORKDIR}/location.txt

RUN python3 -m ipykernel install --user

ENV JUPYTER_TOKEN=openMind


ENV OPENMIND_PERSISTENT_PATH=/home/openmind/volume
ENV BACKUP_PATH="${OPENMIND_PERSISTENT_PATH}_backup"
RUN if [ -d "${WORKDIR}/.cache" ]; then \
        mkdir -p ${BACKUP_PATH} && \
        cp -a ${WORKDIR}/.cache/. ${BACKUP_PATH} && \
        rm -rf ${WORKDIR}/.cache; \
    fi
RUN ln -s ${OPENMIND_PERSISTENT_PATH} ${WORKDIR}/.cache

ENV NOTEBOOK_DIR=${OPENMIND_PERSISTENT_PATH}/notebook/


USER openmind

RUN echo "#!/bin/bash" > ./entrypoint.sh
RUN echo "mkdir -p ${NOTEBOOK_DIR} && mv -f ${CODEDIR}/*.ipynb ${NOTEBOOK_DIR}/" >> ./entrypoint.sh
RUN echo '[ -d "${BACKUP_PATH}" ] && [ -n "$(ls -A ${BACKUP_PATH})" ] && cp -a ${BACKUP_PATH}/. ${OPENMIND_PERSISTENT_PATH}' >> ./entrypoint.sh

RUN chmod +x ./entrypoint.sh
RUN chmod +x ${CODEDIR}/cmd.sh

# no custom command to execute

CMD ./entrypoint.sh ; ${CODEDIR}/cmd.sh
