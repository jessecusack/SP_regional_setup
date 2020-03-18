nx=144; ny=185; nz=88;
hFacC=readbin('grid/hFacC_144x185x90',[nx ny nz]);
for fld={'Theta','Salt'}
    names = dir([fld{1} '*']);
    tmp1=readbin(names.name, [nx ny nz]);
        for k=1:nz
            if any(hFacC(:,:,k)==0, 'all')
                tmp=tmp1(:,:,k);
                tmp(hFacC(:,:,k)==0)=nan;
                tmp=xpolate(tmp);
                tmp1(:,:,k)=tmp;
            end
        end
    writebin(['out/' fld{1} '_filled'], tmp1);
end