#!/bin/bash

docker run --rm -it -v "$PWD":/usr/src/app \
--name scanner --force scanner  \
-e 'ENTRY_URL=http://recipe.hacarus.com/' \
-e 'ALLOW_RULE=/recipe/' \
-e 'IMAGE_XPATH=//*[@id="root"]/div/div/section/div/div/div[1]/figure/img' \
-e 'DOCUMENT_XPATH=//td/text()|//p/text()' \
-e 'PAGE_LIMIT=2000' \
-e 'EXCLUDE_REG=\d(年|月|日|時|分|秒|ｇ|\u4eba|\u672c|cm|ml|g|\u5206\u679a\u5ea6)|hacarusinc|allrightsreserved' \
scanner:latest /usr/src/app/entrypoint.sh
