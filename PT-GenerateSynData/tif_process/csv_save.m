function csv_save(spots,filename)

heading = ["frame" "id" "h_position" "w_position" "size" "intensity" ...
    "velocity" "direction" "velocity_hold_frame"];
writematrix(heading,filename);
col_num = length(fieldnames(spots(1)));  % csv 文件的列数（帧序号 + 原数据列数 - 帧内目标数）
for frame=1:length(spots)
    frame_data = struct2cell(spots(frame));
    spots_num = frame_data{1};
    frame_data{1} = frame*ones(spots_num,1);
    
    data = zeros(spots_num,col_num);
    for col=1:length(frame_data)
        data(:,col) = frame_data{col};
    end
    writematrix(data,filename,'WriteMode','append');
end

end