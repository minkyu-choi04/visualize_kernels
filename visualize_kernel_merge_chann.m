
idx_chan = 3;
sz = size(conv1_k);

slide = ones(sz(3), 5)*0.5;
for c=1:3
    idx_chan = c;
    for i=1:cast(sz(1)/2, 'int32')
        if i==1
            hss1 = squeeze(conv1_k(i, idx_chan, :, :));
        else
            hss1 = cat(2, hss1, slide, squeeze(conv1_k(i, idx_chan, :, :)));
        end
    end
    for i=cast(sz(1)/2, 'int32')+1:sz(1)
        if i==cast(sz(1)/2, 'int32')+1
            hss2 = squeeze(conv1_k(i, idx_chan, :, :));
        else
            hss2 = cat(2, hss2, slide, squeeze(conv1_k(i, idx_chan, :, :)));
        end
    end
    
    slide_h = ones(5, size(hss1,2))*0.5;
    hss = cat(1, hss1, slide_h, hss2);
    
    if c==1
        out = squeeze(hss);
        %out = unsqueze(out, 1);
    else
        out = cat(3, out, hss);
    end
end


figure(22)
imagesc(out);
colorbar;
axis image

