function spots_t = spots_move_out(spots_t,size_frame)
[H,W,~] = deal(size_frame(1),size_frame(2),size_frame(3));
A = bitor(spots_t.h_position < 0,spots_t.h_position > H);
B = bitor(spots_t.w_position < 0,spots_t.w_position > W);
spots_t = spots_remove(spots_t,bitor(A,B));
end