function C=ifft3(A)

C=permute(ifft(permute(ifft(permute(ifft(A),[2 3 1])),[2 3 1])),[2 3 1]);