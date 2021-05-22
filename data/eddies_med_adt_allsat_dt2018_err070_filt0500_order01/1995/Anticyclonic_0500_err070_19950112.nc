CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?Ͼvȴ9X      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P�B�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �8Q�   max       >I�      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Q��R   max       @E�p��
>     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vp          H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�Р          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��1   max       >]/      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�G�   max       B,�v      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B,��      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =iMS   max       C�f      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =~��   max       C�h+      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max                �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       Oí�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��E����   max       ?��Fs���      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       >I�      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Q��R   max       @E�p��
>     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vp          H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q`           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @?   max         @?      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?��t�j~�     �  M�   /   �            6            -      	   )   9   O         d                                     &               
      
                              H         `         O�.P�B�N?t�N|��N�d�P6�zN_I�Nt�NF&�O��OM�N�ăO�W�Oʵ�P���N
�O�>O���OV�O���OF0�N��qOaPN�m�O��O��N�=OI�N��O�[)O tN�,vN�[\OO�N�`�N1��N��O��O4�`O�q8N�p1Nԁ�N-��N.3RN!+\N��O+�:Nc�hNK�>O�&Of��O!ҒN�"�8Q���ě��e`B�49X�D��:�o;o;��
<#�
<49X<D��<D��<T��<u<u<u<�9X<�9X<�j<ě�<���<�h<�h<�h<�=+=+=C�=\)=t�=�P=��=#�
=49X=@�=D��=H�9=P�`=P�`=T��=]/=u=}�=�+=�7L=�t�=��P=���=���=���=��>I�HHGGJO[gt}����{t`[NH5Bw��������tb5+!������������������������������������������������������������&.IPR[hq��z{gUD@/&%qst�����|tqqqqqqqqqq������������������������������������������������
�����qryz}���������zqqqq�����
#4;;;8/#
���`dipz�����������znc`FIEH[������������[NF����������������������
#0<IOOGF0#
�������
#+<LOMG</#
����
#%,,(#
����������115<BLN[agillg[NB651���������������������tstx�������������������������������������������������������214<HUZnz����zna[H<2yyz����������zyyyyyy��������������������#/9<<><1/.#����)0/+*%#���������������������������������� �     yxwx��������������y#*/<EHIH?</$#��������������������������������������������������������������������� ���������������������������/166<BIOUVOCB6//////������������������������������������������������������������YY[^hottwtphb[YYYYYY������������������������������������� 
##%#
HHMTahmnmcaZTHHHHHHH������������������T^^addmz������zwaTST�������������������������������������������������
��#�$���
��������������������6�B�S�U�O�B�$���ìÓ�z�s�uÇéù��������������������������������������������0�=�F�I�M�T�I�=�0�+�.�/�0�0�0�0�0�0�0�0���������������������������������������ؾ����ʾ���	����ؾ������M�A�4�M�Z�f����;�H�O�T�T�T�N�H�;�9�0�5�;�;�;�;�;�;�;�;�Y�e�h�q�e�Y�L�K�L�O�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�����������������������������������������[�h�tĂčĚģĩĩĦĚčā�t�h�\�V�L�E�[�H�U�a�j�n�zÀÈÈÇ�z�n�a�U�I�@�@�<�3�H��#�0�4�<�I�O�I�>�<�0�#����������)�B�N�Q�T�O�C�5�)����������������(�A�N�Z�e�i�e�d�f�Z�N�5�(�!���
���(��(�A�g���������g�Z�5�����������������������������������������������������������������������r�f�D�:�?�M�f������ʾ�	�"�,�:�1�"����ʾ��������������ʿT�`�m�y���������{�y�m�`�T�M�G�=�G�N�T�T�"�.�;�G�T�j�m�w�q�m�`�T�.�"���
���"�b�n�s�{�Ňňł�{�r�n�b�U�H�B�?�B�I�X�b���������������������������������������ҹ������������ùιϹܹ��߹ܹϹʹ��������������������s�k�i�s���������ù������������þûïàÓÆÅÇÍËÓìù�������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͽ�����%�2�4�<�@�4�(��������������������������������������������������׻����ûлڻ���ܻлû�����������x�����Z�f�m�s�z�x�s�m�f�b�Z�M�I�A�:�7�:�A�M�ZƎƚƧƳƺƳưƧƚƎƅƆƎƎƎƎƎƎƎƎ�!�-�:�F�J�F�E�:�1�-�(�!�����!�!�!�!���������ĽƽνӽнȽĽ����������~�y���EEEEEE!EEEEEEEEEEEEEE�zÇÉÇÀ�z�u�n�a�\�a�j�n�v�z�z�z�z�z�z���������������������������������������������������������������������������������	��"�/�;�H�S�T�H�;�/�"��	���������	��"�/�;�F�J�I�@�;�/�"���	������������������ɺպɺº�������������������������"�/�;�G�;�:�/�'�"��	���	������ݿ�����ݿܿӿԿݿݿݿݿݿݿݿݿݿݻ�!�$�,�(�!�����������������#�'�.�'������������������������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D}D{DtDsDvD{D{���ʾ׾۾׾վʾ��������������������������ݿ޿�����ݿܿѿпѿӿݿݿݿݿݿݿݼ��'�@�Y�r��������r�f�Y�M�@�'��	���ŔŭŹ����������������ŹŨŠŕŔŁŃŇŔ¦²¿������������²¦ ¦�����#�!�������� ��������   H L ? ( P 0 ) N C A F  9 2 - . 4 0 8  H y $ k W d Q O > " < 5 ) k � @ 7 O > M K f C M I ! ] F P e -     '  M  q  �  �  v  o  4  r    �  �  �  �  v      H  -    �  �  �  �  �  P  �  �  =    W  �  �  �  �  �  8  1  �  A  �  �  �  `  G  �  r  w  |  P    a  �;o>]/��1�#�
�D��=P�`;��
<e`B<t�=]/=,1<�1=Y�=�\)=�v�<�t�=49X=��#='�=aG�=H�9=t�=D��=t�=,1=u=8Q�=]/=�P=��=�%=,1=D��=�o=]/=q��=m�h=q��=�t�=�hs=ix�=�o=�o=�\)=�hs=��>n�=��=���>2-=��>��>#�
B	B.B��B^B��B�NB��B�MB�B�B��B�tB XB#|B
iB
��B"�B$�.B��B+B��BZKBtBAwB�B!=�B�<B�B!m�B2ZB!lB��B_�B�Be�B,�BDB,�vBy�A�G�B��B*B1�B
�B��B��B-�Bu�B$�QA��#B.?A�m�B��BcxB	@�B��B�B�;B2nB��B��B�B�zB��B��B ?�B:NB�#B
��B"�B%>B�B?�BüB�dB��B@B��B!\�BAtBBzB!�:B�BAB��B@ B!yB@�BI�BEEB,��B�RA�~�B�BB�B<�B�B�WB�RB?rB��B$��A���B?A�zvB;�BC�A�.A�+AtB
�>A�j�AJ��A��?ݛA��A�Q[A��SA�$�A��wA��*A��J@�@�AyAU�Aj��Ac�A�e,A�P�=iMSAE.A��A��rC�fA3iA���@��A>��Bg�@t��A"<>C�j�A��A �(A��`A���A�|:@&��A�~�A}BJ@g��@��mA���C�ǺAO�A}i�@�JA���A�TA2�A���AӂAt8&B
��A�}�AKA���?�H
A�AݎAƉ�A�bA��|A���A���@���@���AU�Ai�Aa��A�}uAГ�=~��AEKAˆCA�}~C�h+A4��A���@�YUA=��B@@s�aA"��C�g�Aǋ�A!��A�\�A��A��G@+n�A�[@A}�@d3Z@ɽ-A�Y�C��dAO}A}_
@���A��]A���A2�g   0   �         	   7            -       
   *   :   O         e      !                              '                                          	         H   	      `               9            5                        !   7      '   %                                                                                                '                                                                                                                                                                        O-t�Oí�N?t�N|��N�d�O��2N_I�Nt�NF&�N�@O]�N�ăOm��O���O�N
�O�G_O�V�N��O<�CO�RNh��N��N�m�O��N���N�=OI�N��O@��O ��N�,vN�[\OO�N�`�N�N��N�O4�`O�q8N�p1Nԁ�N-��N.3RN!+\N��O	`Nc�hNK�>O*��Of��OJN�"  �  =  �  C  O  K  �    .  �  �    R  *  	  t  "  �  �  @  v    i  �  r  �    �  /      W  �  �  �  �  �  �  n  v  �  z  �  �  S  `  ,    	  �  �  a  M�o=�9X�ě��e`B�49X<��
:�o;o;��
<�`B<�C�<D��<�j<�/=]/<u<�1=Y�<ě�<�h<�`B<���=�w<�h<�h=8Q�=+=+=C�=,1=#�
=�P=��=#�
=49X=L��=D��=L��=P�`=P�`=T��=]/=u=}�=�+=�7L=��T=��P=���=���=���>%>I�YQOOQX[gtu����ztgg[Y2039BN[gty|ytg[NB:52������������������������������������������������������������569>BO[ehruuph[OB;65%qst�����|tqqqqqqqqqq������������������������������������������������������qryz}���������zqqqq������
#)/351/#
��jilpz������������zpjb_fot�����������tlgb�����������������������
#0;GHCA90#��
#/<BEDA</
	  
##)*%#
				���������>:98BDN[^fgkjgb[NB>>��������������������~�����������~~~~~~~~����������������������������������������;9<BHTUabhaUH<;;;;;;yyz����������zyyyyyy��������������������#/9<<><1/.#����'&&! ����������������������������������� �     yxwx��������������y#*/<EHIH?</$#��������������������������������������������������������������������� ���������������������������/166<BIOUVOCB6//////������������������������������������������������������������YY[^hottwtphb[YYYYYY����������������������������� ��������� 
##%#
HHMTahmnmcaZTHHHHHHH��������������������T^^addmz������zwaTST�������������������������������������������������
������
�����������������������)�1�5�4�/�!�������������������������������������������������������������0�=�F�I�M�T�I�=�0�+�.�/�0�0�0�0�0�0�0�0���������������������������������������ؾ������ʾѾؾо�����������r�r�v���������;�H�O�T�T�T�N�H�;�9�0�5�;�;�;�;�;�;�;�;�Y�e�h�q�e�Y�L�K�L�O�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�����������������������������������������h�tāčĐĚĝĚęčā�t�l�h�e�a�h�h�h�h�U�W�a�n�z�|ÄÅ�z�n�a�U�O�H�H�F�H�T�U�U��#�0�4�<�I�O�I�>�<�0�#������������5�B�D�J�E�;�5�)����������� ��(�5�A�N�[�`�]�[�Z�X�N�A�5�(������(�A�Z�g�s�x�����s�Z�N�A�=�+�#�!�"�(�2�A������������������������������������������������������������r�f�Y�N�B�E�M�f����ʾ׾�������������׾ʾþ����ɾʿT�`�m�y���������y�y�m�`�T�P�G�S�T�T�T�T�.�;�G�T�`�i�e�`�T�G�;�.�"������"�.�I�U�b�n�y�{Ń��{�n�m�b�U�M�I�E�B�G�I�I���������������������������������������ҹ������ùùʹù��������������������������������������s�k�i�s���������ù������������þûïàÓÆÅÇÍËÓìù����� ������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͽ�����%�2�4�<�@�4�(��������������������������������������������������׻����ûлջֻܻлû����������������������A�M�Z�f�s�x�v�s�h�f�Z�S�M�M�A�=�9�>�A�AƎƚƧƳƺƳưƧƚƎƅƆƎƎƎƎƎƎƎƎ�!�-�:�F�J�F�E�:�1�-�(�!�����!�!�!�!���������ĽƽνӽнȽĽ����������~�y���EEEEEE!EEEEEEEEEEEEEE�zÇÇÇ�~�z�r�n�a�]�a�k�n�x�z�z�z�z�z�z���������������������������������������������������������������������������������	��"�/�;�H�S�T�H�;�/�"��	���������	��"�/�;�F�J�I�@�;�/�"���	������������������ɺպɺº�������������������������"�/�;�G�;�:�/�'�"��	���	������ݿ�����ݿܿӿԿݿݿݿݿݿݿݿݿݿݻ�!�$�,�(�!�����������������#�'�.�'������������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxDxD{D|D�D����ʾ׾۾׾վʾ��������������������������ݿ޿�����ݿܿѿпѿӿݿݿݿݿݿݿݼM�Y�f�r�{�}�u�r�h�f�Y�T�M�A�5�4�2�4�?�MŔŭŹ����������������ŹŨŠŕŔŁŃŇŔ¦²¿����������¿¾²¦¢¦¦�����#�!�������� ��������   L ? ( 0 0 ) N 7 . F # &  - ( % 7 6  W u $ k V d Q O 6 ' < 5 ) k � @ * O > M K f C M I  ] F % e (     n  �  q  �  �    o  4  r  �  7  �  �    r    m    �  �  D  �  ^  �  �  �  �  �  =  �    �  �  �  �  m  8  �  �  A  �  �  �  `  G  �  )  w  |  p    +  �  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  �  	  A  m  �  �  �  �  r  M     �  �  p    �  �  �  V   �  
5  X  �    V  _  '  �    <  1  �  m  �  �  �  .  �  	a  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  C  <  4  -  &          �  �  �  �  �  �  �  w  a  J  4  O  C  7  )      �  �  �  �  �  �  x  L  !  �  �  �  X  *  z  �  :  g  �  �  $  @  K  8    �  �  �  �  d  �  [  �   �  �  �  ~  x  q  j  `  V  L  B  8  ,         �  �  �  �  �    
      
    �  �  �  �  �  �  W  +  �  �  �  v  G    .  ,  +  )  '  &  (  *  ,  .  0  2  4  6  8  ;  ?  B  F  I  �  :  z  �  �  �  �  �  �  �  �  �  �  d  �  _  �  �  �  �    @  e  z  ~  i  >    �  �  u  <  �  �    Z  �  �  j  �    �  �  �  �  �  �  �  �  f  D  $  	  �  �  �  @  �  }      -  =  H  N  R  Q  K  =  "  �  �  d    �  1  �    Z    g  �  �    '  )    �  �  �  k  1  �  �  =  �    H  o  �  9  I  �  �  �  E  �  �  �      �  �  �    �  3  �  �  �  t  p  k  g  c  ^  Z  U  P  K  E  @  ;  4  +  "        �  �  �       "  !      �  �  �  t  F    �  v    �  c  H  
l  2  �     \  x  �  z  h  I    �  W  
�  
  	8  C  �  �  �  �  �  �  �  p  U  :    �  �  �  �  U  X  O    �  ]  �  �    1  <  @  @  =  9  2  &      �  �  �  ;  �  �  <  �  n  l  t  u  v  r  l  d  Y  M  7    �  �  }  C    �  v      �    '  L  ]  `  R  B  .      �  �  �  �  r  K  "  �  �  v  �  �  D      F  �  I  U  i  }  <  �  �  �  �  �  X    �  �  �  �  �  �  �  �  �  �  �  �  �  x  ]  +  �  �  p  (  r  F    �  �  �    �  �  m  :    �  �  W    �  �  �  �  �  �    +  /  !    #  w  �  �  v  9  F  �  T  �  F  �      �  �  �  �  �  �  �  �  �  �  �  ~  N    �  �  Z    �  �  �  �  �  �  �  h  K  '  �  �  �  ~  W  "  �  �  F  �   �  /  )  $          	        �  �  �  �  �  �        �  �             �  �  �  c  '  �  �  @  �  �  S  W  J  �      
    �  �  �  �  t  ;  �  �  �  �  `  )  �  �  o  W  O  H  @  8  /  &        �  �  �  �  �  �  �    o  _  �  �  �  �  �  �  �  �  y  k  [  H  2    �  �  �  �  �  �  �  |  u  k  ]  O  ?  ,    �  �  �  i  :    �  �  (  �  O  �  t  C    �  �  |  R  ,    �  �  }  J    �  b    �  I  �  �  �  �  �  �  �  �  �  	   	  	G  	�  
C  
�    s  �  J  �  �  �  �  �  �  �  �  {  m  \  I  5      �  �  �  }  U  -  �  �  �  �  �  �  �  �  �  ~  r  e  V  F  4  "    �      n  H     �  �  �  z  N    �  �  i  "  �  �  <  �  �  3  �  v  n  b  R  @  )    �  �  �  �  s  [  :    �  �  w    r  �  �  �  �  �  z  j  Z  L  @  5  )      �  �  �  �  e  =  z  u  p  i  c  b  _  X  P  D  5  "    �  �  �  �  o  x  �  �  �  �  �  �  |  e  N  7       �  �  �  �  ~  [  9     �  �  �  �  �  �  �  �  y  n  c  ?    �  �  �  �  m  T  :     S  P  N  K  F  4  !      �  �  �  �     B  d  b  V  J  ?  `  S  D  2       �  �  �  y  7  �  �  9  �       �  t  &  �    (  +      �  �  x    �  �  !  L  U  
D  	  �  �  h            �  
                 �  �  m    ^  �  	  �  �  �  �  �  �  {  h  U  ?  '    �  �  �  �  S  #   �  
  
b  
�  
�    G  i    �  ~  2  
�  
u  
  	�  	  o    O  8  �  j  9  .    �  �  �  �  �  �  �  t  B    �  �  ~  �  �  8  T  `  ]  N  7    �  �  z  @    �  ~  6  �  �  �  �    M  ,  	  �  �  �  p  7    �  �  d    �  g    �  .  �  "