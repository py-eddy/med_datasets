CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�fffffg      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�H      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��w   max       =�j      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @E�ffffg     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vO33334     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2�        max       @R            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�`          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >J��      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�zN   max       B+�}      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�5   max       C�e
      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C�g�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P2>�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?�B�����      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       =�
=      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�ffffg     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vO33334     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @R            x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�,�          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0�   max       ?�B�����     �  T                     j               M   E   
      8   4                     �                                 
            A            
                        )            	   c         Z   O���Nb�'N=6N�ܰNC(FP	u|P�HN��N��kN��N��nP.b�P�G�N��N�wWP?��P9�N�A�O�O��NK,~N���O�N�PvjN{,N���N�%�OE�\N���O"@�M��N	@�N!2�N��)N
�O��_O~^�N#�*P��N��1O��NN�3N���Nhm�NnN��DO���O ��N�LIOq��O	��N�l�O� O@p�NҩuO� N���N�ׅO��ND����w��h���ͼ�t��#�
�D���o��o��o��o;D��;�o<#�
<#�
<D��<D��<T��<e`B<e`B<���<��
<��
<�9X<�j<ě�<���<���<���<�`B<�<�=o=o=C�=\)=\)=�w=#�
='�='�=0 �=0 �=8Q�=<j=<j=D��=L��=P�`=T��=Y�=y�#=��=��=�hs=�t�=���=���=��T=�Q�=�jZWVTS[gt�������xtg_Z!#)06<?<;0(#����������������������������������������zy}�������zzzzzzzzzz@@EQ^bhu��������thO@�����5NXgt�q<)��[Y[\bgt����~vtpg[[[����������������������������������������t�����������������tt�����!������������5NhojaNB5����#0<CC=<0)#	
##'*-+&#
kj�����������xk:@[ht��������yxgd[B:hchty�������ztsrohhh����
#$%'&#
����	/0;G;7/*"	��KNV[cgtt|tg[[NKKKKKK���������������������������$24/+#����4--5639Bg��������xB44,5BMNONB54444444444
)15:5.)%





RIUZanz����}znka^URR����������������<BN[gnjg[[NNDB<<<<<<('&0<HU\nz|znaUH<6/(zvzzz{}����zzzzzzzzz#(/<CB</#�������������UUNJH</&%&&/<FHUUUUU��������������������VWZ`ht�����������thV��������$*)"���������������������������)6:=<:95)����������������(/>IO[WB5)������������������������
#&# 
�������������������������97<DHNSLH<9999999999::;DHJT\adfcaTHD<;::���)39:85)����-((-/1:<BHUWVTPK></-�����������������}|�����������������}�)667761)�.,/02<HNHHHJJH</....	
)6:664-)		#(15772*#
	lqtqpmmnz{������{zmlz|����������������zz��������������������!#-/11/#��������
��������������������������nÇÓàìôøóøìàÓÅ�n�a�U�O�X�a�n���������������������������������������������ùɹϹѹϹù������������������������N�P�Z�b�Z�X�N�A�5�(�"�(�4�5�A�L�N�N�N�N�[�g�t�v�u�t�g�[�U�U�[�[�[�[�[�[�[�[�[�[��(�4�A�f�����s�Z�A�(����������ƚƧƳ������%�����ƭƙƁ�\�:�6�K�hƚ�������������������������������������������������������ܹ������軑��������������������������������������²¼¿����������¿²¦¢¦«²²������������������ìà��s�p�s�~Óù�ҿ��Ŀݿ��1�?�;�(���ѿ������������������������������r�r�f�r�x������������!�(�2�)�(��������������������ϼݼ߼ټۼؼڼּ�������u�o�s�������ܻ��&�.�)����ܻû��~�r�x�����û̻һܻS�_�f�j�_�V�S�F�<�:�2�8�-�"�-�:�F�R�S�S�������!�(�3�(��������������H�T�a�f�j�m�m�h�a�[�T�L�H�@�;�8�6�;�G�H�m�r�y�|���������y�q�p�m�m�f�m�m�m�m�m�mÇÈÓàìù��ûùìàÓÈÈÇÆÇÇÇÇ�H�T�a�e�j�n�j�a�T�H�/�����$�/�6�<�H�6�B�[āĚĦĳĶĬĠĐā�[�B�6����,�6��������������������������������������f�n�r�m�f�b�Z�R�M�K�M�R�Z�`�f�f�f�f�f�f�n�zÄÇÉÎÏËÇÃ�z�n�l�b�a�`�a�k�n�n������������ �����ڹù��ùϹع�ݽ����ݽн̽ĽĽĽĽнٽݽݽݽݽݽ���������������������������������������������������������������������꾌�������������������������������������������������������������������������������m�l�`�T�Q�G�J�T�`�m�y�����y�u�m�m�m�m�m¿������������¿²¨²»¿¿¿¿¿¿¿¿��"�.�;�F�Q�U�W�Q�P�S�G�=�"�����������ʾ׾����������׾ʾ�����������ŔŔŠšŠŞŔňŇŅŇőŔŔŔŔŔŔŔŔ�#�I�_�r�b�^�R�<�0�������������������#�F�S�_�l�x�}�z�x�l�_�S�F�:�-�#�+�-�:�<�F�	��/�C�R�V�P�H�/��	�����������������	�(�1�5�A�N�Z�f�_�Z�N�A�5�/�(����'�(�(�����������������������x�s�o�s�w������������������������������������������������EEEEEEEEEEEEEEEEEEEE���"�(�)�(�#�����������������	��"�-�1�/�)�"��	�������������������	���)�B�L�N�[�c�[�Y�R�N�B�5�)������	��"�.�2�.�.�"�!��	���������	�	�	�	�M�Z�f�s�|���������s�f�Z�M�E�=�=�A�G�M�f�o�r�|��������r�f�d�Y�P�M�M�P�Y�Z�f¥¦²·¸²¦¦¤�e�r�~���������������������~�r�l�e�d�d�e����*�6�C�M�G�C�6�*������������ �ŔŠŭŹ����������������ŹŭŠŝŔňŐŔ�����������������r�f�M�@�;�;�C�M�Z�n����нĽĽ����������������½Ľǽннннн�ǔǡǭǮǭǭǡǞǔǈ�~ǁǈǑǔǔǔǔǔǔD�D�D�D�D�D�D�D�D�D�D�D�D{DqDiDmDoD{D�D�������!�!�!��������������������� ; L O e % : . - 2 a 7 / E ; L [ J p 9 U � B n * L P P E L q t � C 0 � K W O J u Y c > J 7 3 8 D < = 7 B J ? n 6 Y X @ Y  @  �  l  �  V  �  c    �  r  !  	  �  �  #  �  6  �  K  �  �  �  �    ;  �  #  �  �  �  9  �  <    P  j  3  U  �  G  b  �  �  �  .    Z  �  �  �  9  �  @  �  W    �  �    l�D��������t��49X�ě�<�9X=��<49X<e`B;�o<�t�=��T=��w<��
<���=�7L=�o<���<�/=\)<���=��=,1>J��<�h=C�=�P='�=o=y�#=+=\)=��=P�`=49X=m�h=q��=<j=�
==T��=�C�=Y�=aG�=H�9=Y�=}�=��P=��=}�=�+=��=���=�1=Ƨ�=��T>6E�=�1=�->8Q�=���B	�VB%�;BByGB
�+B+eB�B	��B�B"��B��BN�B�aB%�6Bs"B!'�B�IB*�B$s�A��B	7B!��BaRB	�B��BinB_^B[�Bw�B�Bq�BTB�8B��B:�B��B�,B��B� B��B|�B�B�B��B�A�UB. B��B#-(B�mB��B�B��B_A�y�Bo�B,�B��B�?B)�bB	�oB%�yB�BA�B
ZB@�B@�B	E,B��B"�bB ?BA�BMB%�_BR�B >�B��B�'B$�	A�zNB	I;B"?&B��BǇB�BAGB��BDB�EB�9B`�BN�B�4B��B?�B��B�BB6uB��BJB�@B��B�(B��A�u[B��B�B#*�B@vBöB�tB��BM�A���BBB+�}B�>B��B)�KAɐr@�]=�5A��A�JeA8��B�A���?:�b@��kA���A��wA�n�@�'�A��@�ڰ@��f@�}�A1E�A�wzAlgA˼A��{A�+�A�MA@�A�aH?�rA*�cAҚ}A�ѿAG�@��-Ai�A��Aa��AQ�CA�gFA�A@��[A�3-A�KA�k�A�$�C�e
A�,�A�WA��|A\�A@�@�d�A��@	+VA�y�A��m@��A%�BppC��|@^O�A�~�@�NI=��A�p�A�C
A5�B$iA���?N��@��A���A���A��P@�(eA��@�V�@���@�A1�$A���Al�A˄�A�~*A�x�A���A?c�Aȃ�?,�A*a�AҀHA��AH�v@�KAi�A���Aa?AS �A�~�A��@��A��A��A�L�A���C�g�A�]DA�y_A�y�A\)QAA�@��(A�r�@�A��jA���@��A% BA$C��}@cċ                     j               N   E         8   4                     �      	                            
            A                                    )            
   d         [                     +   C               +   7         -   5                  !   5                                       !      +      )                                             $                              +   '                           )   5                     !                                       !      '      '                                                         OICwNb�'N=6N`�NC(FP	u|P.N��N���N��N�e�O��O�UBN�N�X:P�gP2>�NA�)O�N���NK,~N��zO���O��eN{,N���N��-N�,�N���Ny��M��N	@�N!2�N��)N
�O�D�O~^�N#�*Pi[N��1O�ռN�3N���Nhm�NnN��DO���O ��N�LIOq��O	��N�l�O� OdfN�!�O�d3N���N�ׅOQiND��  3  �      �  �  u  �  @  }  �  	  >  H  ]  �  �  {  U  i  �    �  r  z  _  �  �  �  *  8  9  �  J  �  �  
    �  c  '  �  �  �  �    s  6  I  �  
>  �  �    �    r  p  �  ��+��h���ͼ�C��#�
�D��=49X��o:�o��o;ě�<��=@�<e`B<T��<�1<e`B<u<e`B<�1<��
<�1<�j=��
<ě�<���<�`B<�<�`B=,1<�=o=o=C�=\)=t�=�w=#�
=0 �='�=49X=0 �=8Q�=<j=<j=D��=L��=P�`=T��=Y�=y�#=��=��=��P=��=ȴ9=���=��T=�
==�jZ[\bgt�������{tge^\Z!#)06<?<;0(#����������������������������������������zy}�������zzzzzzzzzz@@EQ^bhu��������thO@5BN^_ZQD5)[Y[\bgt����~vtpg[[[������������������������������������������������������������������������ )5@INNKDB5)!#%0<>@<20/#!!!!!!!!
 #'),*%#
ry���������������ur:@Q[ht���������he[B:ehtw����ztsqkheeeeee����
#$%'&#
���	"/;;;3/'"	KNV[cgtt|tg[[NKKKKKK���������������������������#12/#�����A;<DN[g���������tgNA4,5BMNONB54444444444
)15:5.)%





LU_anyzzxnaVULLLLLL��������������������<BN[gnjg[[NNDB<<<<<<./0<HLUUUKH<3/......zvzzz{}����zzzzzzzzz#(/<CB</#�������������UUNJH</&%&&/<FHUUUUU��������������������X[aht�����������th\X��������$*)"�����������������������������)59<<98)���������������&).=HNYUB5)������������������������
#&# 
�������������������������97<DHNSLH<9999999999::;DHJT\adfcaTHD<;::���)39:85)����-((-/1:<BHUWVTPK></-�����������������}|�����������������}�)667761)�.,/02<HNHHHJJH</....	
)6:664-)	
#$./233-%#
	ssurqz������zzssssss����������������������������������������!#-/11/#��������
�������������������������ÇÓàìïîðìàÓÇ�z�k�a�W�_�a�n�zÇ���������������������������������������������ùɹϹѹϹù������������������������A�N�Z�a�Z�U�N�A�5�-�5�7�A�A�A�A�A�A�A�A�[�g�t�v�u�t�g�[�U�U�[�[�[�[�[�[�[�[�[�[��(�4�A�f�����s�Z�A�(����������ƚƧ������������ƳƚƎƁ�u�l�a�b�h�uƁƚ���������������������������������������������������������������軑��������������������������������������²´¿����������¿²§¦ ¦±²²ù������������������ùìÕÌÊÏÓàìù�ݿ������!�!�������ݿٿοʿ̿ѿݼ������������w�r�p�r�}�������������(�/�(�'��������������������ҼټԼӼϼʼ��������~�x�x�|���������ܻ���%�-�'����ܻû����u�t���ûͻӻܻ_�d�f�_�S�S�F�<�:�2�:�F�S�W�_�_�_�_�_�_�������!�(�3�(��������������;�H�T�a�d�i�j�i�a�a�a�T�P�H�D�;�:�9�;�;�m�r�y�|���������y�q�p�m�m�f�m�m�m�m�m�mÓàìùÿûùìàÓÉÉÓÓÓÓÓÓÓÓ�H�T�a�d�i�l�i�a�T�H�/�"���%�0�3�8�>�H�B�O�hāĒĘęčĀ�t�h�[�O�B�6�1�1�3�9�B��������������������������������������f�n�r�m�f�b�Z�R�M�K�M�R�Z�`�f�f�f�f�f�f�zÁÇÇÌÇÅ�z�q�n�f�b�n�n�z�z�z�z�z�z��������������ܹϹùϹϹܹ���ݽ����ݽн̽ĽĽĽĽнٽݽݽݽݽݽ���� ������������������������������������������������������������������꾌�������������������������������������������������������������������������������m�l�`�T�Q�G�J�T�`�m�y�����y�u�m�m�m�m�m¿������������¿²¨²»¿¿¿¿¿¿¿¿�"�.�;�E�O�T�V�P�N�J�E�"��������"�����ʾ׾����������׾ʾ�����������ŔŔŠšŠŞŔňŇŅŇőŔŔŔŔŔŔŔŔ�
�#�<�I�Z�l�b�\�P�<�0����������������
�F�S�_�l�x�}�z�x�l�_�S�F�:�-�#�+�-�:�<�F�	��/�B�Q�T�H�A�/��	�����������������	�(�1�5�A�N�Z�f�_�Z�N�A�5�/�(����'�(�(�����������������������x�s�o�s�w������������������������������������������������EEEEEEEEEEEEEEEEEEEE���"�(�)�(�#�����������������	��"�-�1�/�)�"��	�������������������	���)�B�L�N�[�c�[�Y�R�N�B�5�)������	��"�.�2�.�.�"�!��	���������	�	�	�	�M�Z�f�s�|���������s�f�Z�M�E�=�=�A�G�M�f�o�r�|��������r�f�d�Y�P�M�M�P�Y�Z�f¥¦²·¸²¦¦¤�e�r�~���������������������~�r�l�e�d�d�e���*�6�C�I�D�C�6�1�*�������������ŔŠŭŹ��������ŹŭŠŞŔŒŔŔŔŔŔŔ�r�������������������r�f�Y�G�C�C�M�i�r�нĽĽ����������������½Ľǽннннн�ǔǡǭǮǭǭǡǞǔǈ�~ǁǈǑǔǔǔǔǔǔD�D�D�D�D�D�D�D�D�D�D�D�D{DvDnDrD{D�D�D�������!�!�!��������������������� ; L O T % : 5 - 2 a 8 $   < I Y J f 9 P � 8 j  L P P D L . t � C 0 � G W O G u \ c > J 7 3 8 D < = 7 B J D f & Y X A Y  �  �  l  l  V  �  �    �  r  �  �  D  =      -  �  K  ?  �  �  �  1  ;  �  �    �  �  9  �  <    P  4  3  U  �  G  A  �  �  �  .    Z  �  �  �  9  �  @  l  �  e  �  �  �  l  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  �    #  1  /      �  �  �  R    �  w    �  4  �  X  �  �  �  �  }  y  u  m  b  W  M  B  7  -  $      	      �            �  �  �  �  �  �  �  �  t  n  w  �  h  @                      �  �  �  �  m  C    �  �  �  Q  �  |  x  t  p  j  `  V  L  A  6  *        �  �  �  �  �  �  �  �  �  �  �  m  Z  H  7  #    )  2  4  /  $  #  '  �  �  L  �  Z  �  F  i  p  q  j  4  �  �  d  I  �  I  h    �  �  �  �  �  �  �  �  �  �  �  �  z  l  `  X  %  �  �    Q  ?  @  @  =  4  )      �  �  �  u  H    �  �  E  �  �  '  }  �  �  �  �  �  �  �  �  �  �  �  �  p  ]  I  5    
  �  �  �  �  �  �  �  �  �  �  z  *  �  S  +  ?  8  -         %  q  �  �  �  	  	  	  �  �  �  ^    �  C  �  �  �  K  X  W  �  �  �  �  �    %  3  :  =  :  +    �  h  �  5  O  �      %  /  8  ?  E  G  G  F  A  :  1  '      �  �  �  �  ]  ]  ]  Z  V  N  C  6  %    �  �  �  �  k  6  �  �  M  �  d  �  �  �  �  �  �  �  w  :  	    �  �  �  �  �  w  +  �  �  �  �  �  v  [  I  D  /  #  F  D  @    �  �  �  �  �  �  l  p  t  x  {  }    �  �  �  �  �  �  �  �  �  �    8  V  U  T  N  D  8  +      �  �  �  �  p  H    �  �  ~  G    7  O  b  g  b  Y  M  A  5  '      �  �  �  p  (  �  �  5  �  �  �  �  �  �  �  �  m  R  7       �  �  �  n  2   �   �  �     �  �  �  �  �  �  �  a  7    �  |  2  �  �  =  �  �  �  �  �  �  �  ~  V  *    �  �  �  M    �  �  x  P    �  -  �  �  �  %  ]  q  k  W  C     �  �    i  �  �  
B    �  z  k  \  M  >  0  !      �  �  �  �  �  �  �  �  �  �  �  _  ^  \  Y  V  R  K  D  ;  0  *  )  '  $  !      �  �  A  �  �  �  �  �  �  {  \  5    �  �  �  r  P  +    �  �  �  �  �  �  �  �  �  �  �  v  g  Y  L  <  #  �  �  �  5  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        !  ,  -  �  �  �  �  �    !  $    �  �  �  .  �  ]  �  j  �  j  8  ;  ?  B  E  H  K  L  L  M  M  M  M  O  T  Y  ^  c  g  l  9  /  $        �  �  �  �  �  �  �        .  =  K  Z  �  �  �  y  ]  ?  !  	  �  �  �  �  �  �  �  �  �  �  �  �  J  <  <  9  4  +        �  �  �  �  s  B    �  p    �  �  �  �  �  �  �  s  V  8    �  �  �  �  s  M  "  �  �  u  �  �  �  �  �  �  �  �  �  d  I  0    �  �  �  Q  �  t   �  
  �  �  �  �  �  �  p  P  -      �  �  �  u    �  7   �            �  �  �  �  �  q  V  ;      �  �  �  �  c  �  �  �  �  �  �  �  w  d  K  (  �  �  y  $  �  B  �  Y  �  c  c  c  `  W  G  :  -       �  �  �  �  �  �  �  �  �  O    &        �  �  �  q  :  �  �  t  /  �  �  �  j  m  Q  �  �  �  �  �  d  D     �  �  �  ]    �  m    �  �  S    �  �  �  �  �  �  �  �  q  ]  G  -    �  �  s  @    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  j  \  N  �  �  �  �  v  ]  B  $    �  �  �  �  t  X  <       �  �      �  �  �  �  �  ^  8    �  �  ~  J    �  �  i  C  B  s  j  c  `  d  Z  J  6    �  �  �  |  K    �  �  5  �   �  6  &        �  �  �  �  |  W  0    �  �  M    �  �  �  I  A  8  *        �  �  �  �  �  �  t  [  @  '    �  �  �  �  }  v  n  g  _  O  7    �  �  �  �  d  B  $    �  �  
>  
8  
1  
  	�  	�  	�  	M  	  �  �  l    �  =  �  �  �    �  �  z  ?    �  v  6  �  �  �  |  T  :    �  �  R  �  �  7  �  �  �  �  �  �  k  D    �  �  �  v  J    �    �  �  l  �          �  �  �  �  �  j  M  .    �  �  �  �  �  �  /  W      y  g  S  A  0  "      �  �  �  �  �  [  �  �    �  �      �  �  �  w  "  �  b  
�  
q  	�  �    �  4  L  r  b  R  B  1      �  �  �  �  �  u  Z  ?  $   �   �   �   g  p  T  9    
  �  �  �  �  �  �  v  Z  @  +       �  �  �  �  �  �  �  �  �  �  J  �  �  �  H  �  �    A  k  
M  	
  Y  �  �  |  e  @    �  �  �  �  Z  6  &    �  �  �  z  M   