
#!/usr/bin/env bash
# clean docs and doctrees
rm -r docs/*
rm -r doctrees/*
# make the documentation in gitignore folder
cp examples/README.rst sphinxdocs/source/examples.rst
cd sphinxdocs
make clean
make html
# copy sphinx build into docs and doctrees
cp -r build/doctrees/* ../doctrees/
cp -r build/html/* ../docs/
# grep -rl -e "word-break: break-word" --exclude=sphinx_build_script.sh | xargs sed -i 's/word-break: break-word/word-break: break-all/g'
cd ..
sed -i 's/word-break: break-word/word-break: break-all/g' docs/_static/basic.css 
