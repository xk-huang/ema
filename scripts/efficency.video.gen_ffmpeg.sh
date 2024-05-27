base_dir=$1
find $base_dir -type d -name 'novel_view' -print0 | 
    while IFS= read -r -d '' line; do 
        echo "$line"
        vid_id="$(echo $line | cut -d/ -f3)"
        vid_type="$(echo $line | cut -d/ -f4)"
        echo vid_id: $vid_id
        echo vid_type: $vid_type
        if [[ "$vid_type" == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]" ]]; then
            echo "full_view"
            mv "$base_dir/$vid_id/$vid_type" "$base_dir/$vid_id/full_view"
        elif [[ "$vid_type" == "[0, 6, 12, 18]" ]]; then
            echo "train_view"
            mv "$base_dir/$vid_id/$vid_type" "$base_dir/$vid_id/train_view"
        else
            echo "unknown"
        fi
    done

echo "" > ./tmp/ffmpeg.sh
find $base_dir -type d -name 'novel_view' -print0 | 
    while IFS= read -r -d '' line; do 
        # echo "$line"
        vid_id="$(echo $line | cut -d/ -f3)"
        vid_type="$(echo $line | cut -d/ -f4)"
        vid_date="$(echo $line | cut -d/ -f5)"
        vid_date2="$(echo $line | cut -d/ -f6)"
        echo ffmpeg -y -framerate 30 -pattern_type glob -i \'"$line/*.png"\' -b:v 4M -c:v libx264 -pix_fmt yuv420p -filter:v "crop=512:512:0:0" \'"$base_dir/$vid_id.$vid_type.$vid_date.$vid_date2.mp4"\' >> ./tmp/ffmpeg.sh
    done

