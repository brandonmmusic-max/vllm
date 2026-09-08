# Build the current PR using docker/Dockerfile first. Never overlay new Python
# onto RC5's old vLLM extension binaries and call that a Jovian port.
ARG JOVIAN_IMAGE
FROM ${JOVIAN_IMAGE}
RUN uv pip install --system --no-deps \
      "b12x @ https://github.com/brandonmmusic-max/b12x/archive/ac8ef2ca23ab1f5bb45a94976b36a3ded0006bf1.tar.gz"
COPY examples/trellismx/serve.sh /opt/trellismx/serve.sh
ENTRYPOINT ["/bin/bash", "/opt/trellismx/serve.sh"]
