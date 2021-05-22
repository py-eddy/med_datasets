CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�vȴ9X      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��r   max       P��y      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �u   max       =�G�      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @FH�\)            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @vh             (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @M@           h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;D��   max       >��j      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B1^�      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B1{      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @1   max       C�q      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @|/   max       C�;      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         `      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��r   max       Pl��      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?�;�5�Xz      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       > Ĝ      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @FH�\)        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @vg��Q�        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @I            h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�q�          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���$tS�   max       ?�:)�y��     @  L�           _                        	            H   �               8         (   '      $      "         &         #   �      -      
   ,         
   ^   	      	   -   B   O��lNjO��P��yO�BFOE�\O#��O��,P+�jO��O4U�N��M��rO�8O|�Pq~TPQH�OM�ON�N��^OF��O�$�O�tbN_�O�4�O1�ON���P"p�O0?�O�VcO�7�N_~�O�=O?|O���O���O�ooO�KPl��NF:�N��9O�;FN�YO�EvN���P7ClN�9Ng�N��O���Oa��M���u��o;o;o;D��;�o<T��<T��<T��<T��<�o<�t�<��
<�9X<���<���<���<�/<�`B<�`B<�<�=o=o=o=+=C�=\)=\)=��=�w='�=0 �=49X=<j=<j=@�=@�=@�=@�=D��=ix�=�%=�%=�%=�o=��
=��=�Q�=ě�=��=�G�#0<U\bed^UB<0#������������������������)550/)�����-+/:B[g��������tgN<-UUZ_gt�����������g[U���������������������� 	"/0556;/"	��5;BFLIB:5)���
#**0,01#
������745ADIUnqpsvrrlbUH<7)55BDIHBB;5)%��������������������YZ[glt{tg[YYYYYYYYYY��������������������)58>=BEB50)#5L�������]B5)"$ &6BO[ht�����hO6$wttwyz������������zwGEF=??HUamga^\WaUHGG��������������������kkntz���������{zwnmk�������
�������mjint�������������tm��������������������������
/<BDD</#	��������������������������
������)")BO[hqx}�zj\[OB8)�����������������������)+.,'������������
#+,*$"
��=COOP\hihh\ODC======}}����������������}����������������������)BBFEB<5)��=A>?ENZg�������t[MB=��������
 "
�����moyz������������zqnm������/31)����������������������������������������#
�������
 #$$"#beggmz~~{zzmbbbbbbbb���)2410+#�978ABN[[^_[ZNB999999�������)=BA�������)6@960)'IKIHA<41/./<HJIIIIII#,/00/+#"#######MMHDBGQam}����zv_VTM������
#&&!
��������������������������������Ż����������x�j�S�J�C�D�S�_�x��������������������ù������������������������	���������������������������������6�O�Y�n�s�t�n�[�B�6�����úõ��������/�H�T�d�d�[�T�N�H�;�/�"��������������û˻лۻлƻ����������������������T�W�a�g�m�q�o�m�a�T�H�;�6�8�;�<�D�H�Q�T����	��������׾Ǿ����������ʾ׾���Z�^�x�����~�g�Z�N�5��������
�(�5�Z�r���������������������r�f�Y�J�C�Y�h�r�h�uƁƎƖƔƎƎƅƁ�u�h�\�Z�O�H�H�O�\�h�A�N�P�W�Z�[�_�b�Z�W�N�A�A�8�5�5�5�?�A�A�����������������������������������������A�M�Z�`�\�Z�T�M�A�4�/�(�����(�4�7�A�Z�f�g�k�p�o�j�f�Z�M�D�A�8�=�A�B�E�I�M�Z�N�g�t�t�[�B�5�#�����������N���ûл��'�=�H�K�I�D�4����û�������������������������������������������������ù���������������������������ùõøùù�������������������r�n�f�Y�S�Y�f�r�{�ƚƧƳ����������ƳưƧƚƐ�~�u�q�uƁƇƚ����������� ����������ƳƧƜƗƘƟƳ����(�4�A�G�Y�_�b�]�T�M�A�4�#���
��������������������������������������������.�G�T�m����������o�`�G�;����!�"�(�.àù����������ùðìàÖÓÇÁ�~�Ç×à�T�\�`�e�a�`�T�G�C�;�.�-�.�.�6�;�G�T�T�T�ʿ	���!�"����	��׾������}�v�����ʾ�(�4�A�A�K�O�M�H�A�<�4�(���������������
������
������������������M�Z�f�s�{������������s�f�Z�M�A�5�9�A�M�`�i�m�m�m�k�a�`�T�T�L�L�T�V�`�`�`�`�`�`ÇÓàù����������������ìàÓÌÉÇ�{Ç�(�5�:�A�G�F�D�A�6�(������	����(�����#�(�5�2�(��� �����������A�N�Z�g�������������������s�m�X�F�A�5�AD{D�D�D�D�D�D�D�D�D�D�D�D�D�D}DuDqDoDqD{�����������������������������������������"�/�U�`�e�^�=�/�	�������������������	�"�нݽ����ݽннϽĽ��Ľ̽нннннлl�x�����������x�l�_�V�U�_�j�l�l�l�l�l�l�~�w�v����������ɺԺںغֺɺ����������~ŠŭŹ��������ŹŭŠŠŜŠŠŠŠŠŠŠŠ�#�0�<�I�L�R�W�Y�U�I�<�0�#�����������	�#�I�U�b�n�o�s�o�n�b�U�I�I�B�G�I�I�I�I�I�I���������ռ��Ӽʼ��������~�z�v�h�c�r���~�������������������������~�x�s�~�~�~�~¥¦¬²³»·²¥¥¥¥¥¥ǭǡǖǔǈǆǈǊǔǡǡǡǭǭǭǭǭǭǭǭ�B�O�X�hčĚĦĮĮĦĚčā�s�R�O�B�6�)�BEuE�E�E�E�E�E�E�E�E�E�E�E�EuEiEgEiEjEnEu�l�y�y��|�y�s�l�g�`�l�l�l�l�l�l�l�l�l�l . D X   E 9 5 0 9 1 . ` b = R / 6 H d X G $ A G , O M ? 6 -  _ b D ; W , H 7 P ; * ] G " u V U U @  _  �  "  T  L  �  �  r    �    �  �  3  1  n  �  �  �  A  �  �    �  x    �  �  *  �  6  �  �  �  �      f  K  �  Z  �  	  �  G  �  4  %  j  M  �  �  ><u;D��<�o>��j<�9X<��
<�h=#�
=0 �<�`B<�h<�/<���=\)=#�
=Ƨ�>$�/=8Q�=#�
=\)=T��=�{=]/=��=�t�=�t�='�=�\)=ix�=�hs=y�#=8Q�=��T=��=�7L=��T>B�\=q��=�^5=L��=ix�=���=�7L=�v�=��> Ĝ=�E�=�v�=ȴ9>\)>-V=�B&)B��B��B	j�B
q(B"wxA��B�LBh�B'R}B� B��B	A�B	 B��B��B��B WB�"B"�UB '�B��BL�B]CBc�B"@B��B��B"��Bu�B�\B1^�B �B�dBo�B	�EB`B�5B�kB��B^wB$dA��@B��B�BʜB�B�CB��A���B�B+,xB&?B�^B�KB	��B
�B"HA���B�0B��B'@B �B�B	?�B=)B�#B	�B�,B 9�B��B"��B >B�B~ BD�B��B"@bB��BxB"�B�}B��B1{B@�B�$B��B	<�B1FB��B�sB�cB_[B#��A�BB��B@FB��B�B�B�wA��>B��B+IV@��A�MOA���A�m�A��@���A�m�AU�A�Ng@��~B�nA�}A���A9�UA?p1A�}!@�}CA�A�A�=*@��B�Bd�A8��Ar��Af�YA˴Ae�wASRA7��A��"A@�XAh�A�yA�y A�KIA��"C���A�d�A�4�A*<�@���@f�A�$�AꔀA���@���@1A�9$Bz'A�a3C�qA�@@��Aό�A��KA�} A���@�D�A��?AU9A���@��B@sA��A���A;�A?�A��@��A�-Aщs@��BÏB�mA9 �Ar�Ag�A˙gAe��AR��A8�xA�}�AAAhUEA�s�A��A�ZA���C�גA�KA�|�A(�y@���@&oA�"�A�vKA�4@�̍@|/A�i�B��A��oC�;A��           `                        
            H   �               8         )   (      $      "         '         $   �      -      
   ,            ^   
      	   -   C      '         7   !            3                     /   /               %         !         )                        #         3                     1            "         #         !               3                     )                           !         '                        !         3                                 "      O� NjOV�O�sO��O7�N�qOiGKP+�jO��O4U�N��M��rO�8N:7�P=�iO��?O.�N�3GN���O8��Os�KO{N_�O�4�N镜N�g�O�B�O0?�OQVO@@mN_~�O�@O��Os�O��O4�>O ��Pl��NF:�N��9O�;FN�YO�"N���O��N�9Ng�N��O���O6ڲM���  �  �  n  �  �  �  2  N  :  "  �  �  <  g  �  �    �    9  �  e  �  �      ~  �    �  +  [  J  �  -  �  p  �  =  �  �  I  �    �  
z  �  8  �  �  �  e�D����o;�o> Ĝ;ě�;��
<�o<�o<T��<T��<�o<�t�<��
<�9X<��=�P=��-<�h<�<�h<��=L��=��=o=o=,1=\)='�=\)=49X=0 �='�=]/=@�=@�=H�9=\=D��=@�=@�=D��=ix�=�%=�hs=�%=ȴ9=��
=��=�Q�=ě�=�S�=�G� 0<IW`cb\UH<0# ������������������������#12-*)����@AFN[gt�������tg[NC@YX[`dgt����������g[Y��������������������	"'/3230/-"
	)5>BGIGB5) ���
#**0,01#
������745ADIUnqpsvrrlbUH<7)55BDIHBB;5)%��������������������YZ[glt{tg[YYYYYYYYYY��������������������)05))' )5Bg������g[NB5'1014BO[hnsyzzwt[OB61yuuvxz������������zyH@A@HUaiea[XUHHHHHHH��������������������llouz�����������zpml��������������������su|��������������wts��������������������������
/<BDD</#	���������������������������	�������..,+16BO[s{zundOB>;.����������������������'*("���������#''$#
����=COOP\hihh\ODC======�������������������������������������������	)5AEDB;5)�AAEGN[gt������tg[NFA��������

�����snpz|������������zss������/31)����������������������������������������#
�������
 #$$"#beggmz~~{zzmbbbbbbbb�� &(&$!978ABN[[^_[ZNB999999��������
������)6@960)'IKIHA<41/./<HJIIIIII#,/00/+#"#######MMHDBGQam}����zv_VTM�����
 $$#
������������������������x���������������������x�n�_�O�F�M�_�l�x������������������ù��������������������������	����	��������������������������)�6�D�P�U�U�M�B�6�)�����������������/�<�H�T�Z�]�T�O�H�;�/�"��������������û˻л׻лû����������������������T�a�d�k�m�n�m�j�a�T�H�;�;�;�=�@�H�I�T�T�ʾ׾����	����	����׾̾������¾��Z�^�x�����~�g�Z�N�5��������
�(�5�Z�r���������������������r�f�Y�J�C�Y�h�r�h�uƁƎƖƔƎƎƅƁ�u�h�\�Z�O�H�H�O�\�h�A�N�P�W�Z�[�_�b�Z�W�N�A�A�8�5�5�5�?�A�A�����������������������������������������A�M�Z�`�\�Z�T�M�A�4�/�(�����(�4�7�A�M�Z�f�i�j�f�Z�M�M�M�M�M�M�M�M�M�M�M�M�M��5�[�t�~��t�[�N�5�$�������л����%�1�4�/�'������ܻ׻»����Ļ��������������������������������������������������������������������������������Ҽf�r�������������������r�r�f�[�f�f�f�fƚƧƳ����������ƶƳƲƧƚƒƎƀ�uƁƎƚƳ��������������������������ƳƭƨƩƲƳ�(�4�A�M�W�[�T�M�A�4�/�(�������$�(�����������������������������������������.�G�T�m����������o�`�G�;����!�"�(�.àìùù��ÿùìéàÓÇÄÃÇÐÓßàà�;�G�T�`�c�`�_�T�G�?�;�4�0�8�;�;�;�;�;�;�ʾ׾�	��������׾ʾ��������������ʾ�(�4�A�A�K�O�M�H�A�<�4�(��������������
�����
����������������������f�s�����������z�s�f�Z�W�A�=�?�F�M�Z�f�`�i�m�m�m�k�a�`�T�T�L�L�T�V�`�`�`�`�`�`àëù��������������������ùñàØÓÖà��(�5�A�D�D�A�A�5�2�(����������������"�(�3�2�(�������������Z�g�s���������������������s�Z�Q�I�F�N�ZD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D{D{D~D������������������������������������������"�/�U�`�e�^�=�/�	�������������������	�"�нݽ����ݽннϽĽ��Ľ̽нннннлl�x�����������x�l�_�V�U�_�j�l�l�l�l�l�l�~�w�v����������ɺԺںغֺɺ����������~ŠŭŹ��������ŹŭŠŠŜŠŠŠŠŠŠŠŠ��#�0�<�B�I�J�I�I�<�0�#��
������� ���I�U�b�n�o�s�o�n�b�U�I�I�B�G�I�I�I�I�I�I���������ʼӼ׼ռʼż��������������������~�������������������������~�x�s�~�~�~�~¥¦¬²³»·²¥¥¥¥¥¥ǭǡǖǔǈǆǈǊǔǡǡǡǭǭǭǭǭǭǭǭ�B�O�X�hčĚĦĮĮĦĚčā�s�R�O�B�6�)�BE�E�E�E�E�E�E�E�E�E�E�E�E�EEuElEjEnEuE��l�y�y��|�y�s�l�g�`�l�l�l�l�l�l�l�l�l�l * D R  G 6 0 , 9 1 . ` b = @ / , E A T ?  : G , G O 6 6 #  _ d 5 6 M  C 7 P ; * ] @ " M V U U @  _  �  "  �  �  �  �    �  �    �  �  3  1  H  +  �  �  �  �  �  �    x      �  *  �  �  �  �  �  E  �  �  v  ,  �  Z  �  	  �  n  �  |  %  j  M  �  �  >  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  �  �  �  �  �  �    b  :    �  �  {  *    �  �  �  �  �  �  �  �  �  �  y  m  a  Z  S  J  ?  4  #    �  d  j  m  m  i  c  Y  K  7    
  �  �  �  �  U  %  �  �  m    X  �  f  �  d    n  �  �  M  �    �  Z  9  �  |  �  G  �  �  �  �  �    l  Q  M  I  C  ;  2    �  �  �  g  /  �  �  �  �  �  �  �  q  S  2    �  �  �  [  '  �  �  �  �  �    *  0  2  2  0  ,  %       
  �  �  �  �  �  w  Z  1  �  =  I  N  I  @  0    �  �  �  �  Y  ,  �  �  V  �  *  c   S  :  %  �  �  �  �  �  �  �  �  �  i  B    �  �  �  Y       "      �  �  �  �  �  �  p  U  8      �  �  �  x  C    �  �  �  �    q  b  Z  R  H  2    
  �  �  �  �  �  �  l  �  �  �  q  _  I  1    �  �  �  �  t  K    �  �  V  
   �  <  ?  C  F  D  +    �  �  �  �  �  t  V  9    �  �  �  �  g  _  V  M  A  6  *      �  �  �  �  �  �  q  P  )    �    K  l  �  �  �  �  �  �  �  �  �  �  �  Q    �  }  4  �  (  m  �  �  �  �  t  \  E  3    �  �  �  [    �  �  �  i  
�  -  �    f  �  �      �  �  E  �  
�  
I  	�  �  D    �  �  �  �  �    s  `  J  1    �  �  �  T    �  q    �  j  �  �  �    
      �  �  �  �  �  �  �  �  �  �  �  Q  �  8  9  9  6  +         �  �  �  �  �  �  �  �  �  �  �  ~  z  �  �  �  p  Y  @  $    �  �  �  _    �  �  G  �  �  u  �    3  L  Y  _  d  e  a  X  @    �  j    �  $  d  w    _  k  v  ~  �  �  �  �  |  r  f  P  2    �  �  �  E  �  �  �  �  �  �  �  �  �  �  �    d  @    �  �  �  r  D     �    �  �  �  �  �  �  �  z  d  D    �  �  b    �  G  k  #  j  �  �        �  �  �  U    �  t  	  {  �  D  �  �  �  w  z  |  |  v  p  f  Z  M  A  4  (        �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  e  Q  B  6  &    �  �    x    �  �  �  �  �  �  }  V  ,  �  �  �  E  �  �    �     ^  �  �  �  �  �  �  �  d  9    �  �  �  ;  �  �  )  +  �  ?      %  *  )  $      �  �  �  �  z  Q    �  �  g  6    [  M  @  2  %      �  �  �  �  �  �  �  �  �  �    r  e  �  �    /  B  G  H  1    �  �  c  +  �  �  w    �  H    �  �  �  �  �  �  �  v  R  +  �  �  �  Z    �  p  �  M  �    ,  '       	  �  �  �  �  d  0  �  �  �  I    �  \  �  e  r  �  �  �  f  >    �  �  k  ,  �  �  �  C  �  H  R   �  �  3  �  �  !  R  k  i  R    �  ^  �    �  �  P  �  �  	  �  �  �  �  �  v  X  5    �  �  �  g  @  "    �  �  �  _  =    �  �  �  J    �  �  �  �  `  &  �  �  I  �  �    �  �  �  �  �  �  �  �  �  �  w  h  Y  J  ;  -         �   �  �  �  �  {  d  K  2    �  �  �  �  {  [  :    �  �  G  �  I  4  2  /  )        �  �  �  �  W    �  ]  �  o  �  $  �  �  �  �  �  �  �  �  �  u  q  t  w  z  }    �  �  �  �  �  �  �           �  �  �  �  �  t  6  �  x  �  G  P  Z  �  �  �  �  �  p  Z  C  ,    �  �  �  �  ^  7    �  �  �  
-  
J  
Y  
j  
Y  
,  	�  
z  
l  
W  
9  
  	�  	z  	  b  �  �  �  -  �  �  �  �  n  U  <  %    �  �  �  �  �  _  :    �  �  �  8  �  p  �  �  a  ,  �  �  �  �  �  Y  y  �  ~  `  :    �  �  �  �  i  K  -    �  �  �  �  s  S  0    �  �  �  y  R  �  �  �  t  N  &    �  �  ~  F    �  n    �    T  �    �  �  �  �  �  �  �  G    �  g    
�  
  	�  �  F  8  �  i  e  :    �  �  _    �  s  :     �  {  #  �  r  C     �   �