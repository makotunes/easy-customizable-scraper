#!/bin/bash

echo ${ENTRY_URL}

if [ -n "$DOCUMENT_XPATH" ]; then
  DOCUMENT_XPATH=-"-document-xpath ${DOCUMENT_XPATH}"
  echo ${DOCUMENT_XPATH}
fi
if [ -n "$IMAGE_XPATH" ]; then
  IMAGE_XPATH="--image-xpath ${IMAGE_XPATH}"
  echo ${IMAGE_XPATH}
fi
if [ -n "$ALLOW_RULE" ]; then
  ALLOW_RULE="--allow-rule ${ALLOW_RULE}"
  echo ${ALLOW_RULE}
fi
if [ -n "$DENY_RULE" ]; then
  DENY_RULE="--deny-rule ${DENY_RULE}"
  echo ${DENY_RULE}
fi
if [ -n "$PAGE_LIMIT" ]; then
  PAGE_LIMIT="--page-limit ${PAGE_LIMIT}"
  echo ${PAGE_LIMIT}
fi
if [ -n "$EXCLUDE_REG" ]; then
  EXCLUDE_REG="--exclude-reg ${EXCLUDE_REG}"
  echo ${EXCLUDE_REG}
fi

/usr/bin/mongod & python -u /usr/src/app/src/main.py ${ENTRY_URL} \
 ${ALLOW_RULE} ${DENY_RULE} ${PAGE_LIMIT} ${DOCUMENT_XPATH} ${IMAGE_XPATH} ${EXCLUDE_REG}
  


