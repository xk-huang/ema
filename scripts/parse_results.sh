echo "exp_name,subject_id,cfg,"

find "$(realpath --relative-to='.' ${1})" -type f -name 'metrics.txt' -exec sh -c 'dirname $(dirname {})' \;| uniq |
while read -r path; do
    list=($(echo "$path" | tr '/' ' ')) 
    exp_name="${list[1]}"
    subject_id="${list[2]}"
    cfg="$(echo ${list[3]}| tr ',' '-')"

    echo -ne "$exp_name,$subject_id,$cfg"
    find "$path" -type f -name 'metrics.txt'|
    while read -r file; do
        file_parse_list=($(echo "$file" | tr '/' ' ')) 
        metric_name="${file_parse_list[-2]}"
        echo -ne ",${metric_name},$(tail -n1 $file | awk '{print $3}')"
    done

    echo ""
done