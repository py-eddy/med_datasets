CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�|�hr�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�$q   max       P�͇      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�v�      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E�z�G�     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vffffff     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P`           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�Р          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       >�$�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�r�   max       B,�'      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{b   max       B,��      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�>�   max       C�8      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�ǉ   max       C�D      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         o      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�$q   max       Pl�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Q�   max       ?�b��}W      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >?|�      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @E��Q�     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vfz�G�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P`           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�=�          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��)^�	   max       ?�|�Q�     �  QX   
            /         
      "   -      E  o         4            4                                 $            !         
   \   I   R          
                      w               	O�N��LN�r�N�mO��N�^�N2gN��2N�cO��rP��N�2�P	YP�͇OxT�N���P{��O�Oe�lO9͈O��lN�'N��,N"��O���N�W$N4jNL�SON��O�#Nt_kO��PNL�lM�$qOA�O�+�O�� O���N�K�PT�O�adP*�~N	/DO��:N0��O�:Om8;OBB�NQ6<N��O]��O��O*S�O�YoO*ZeN"E�NEV���㼃o��`B�ě��D��;�`B<49X<u<u<�o<�C�<�C�<���<�1<�1<�9X<�9X<ě�<���<�/<�/<�h<��<��=o=o=o=\)=\)=#�
=#�
=#�
='�='�=,1=0 �=49X=49X=49X=8Q�=8Q�=@�=H�9=P�`=T��=e`B=e`B=ix�=ix�=��=��=�+=�C�=�\)=�{=�Q�=�v�rtw��������������|tr\\bht����vth\\\\\\\\"/693/.%"##(-0<IKRTSNI<10)#�������
�����������������������������E?>@EHKQQNJHEEEEEEEE9408<HPQUZYUH<999999����������� ���������5BN[^gmg[5��&"!$)5g��������g[NB&��������	
���������ejinv������������zme����gN?2*)2N[g��������������������������{z}�����������{{{{{{��)w�|nlph[KD0����������������������������������������������������������&BN[gt~����tg[NB51+&'�����������������������������������������������������������������������������������

���������������������������������
#/<DIHD</# ���������

�����##./2/+# MDCFOht��������th]YM��������������������"#(0120#%)/59BHFFEB@75)������������ �������������������������������������������������������������������������337:82)��&$%,-6BO[hheeb[WB6)&WWn��������������oeW-/1<AHLH=<:/--------�������		��������������������������������������
##
������������ %)��
)412220)31356BNQNNB>75333333

!)+/-)





yyvsnz������������|y������

�������
!)15=>52):??EJZTXVOD6*����!''!�������������������������##,///#Ŀ��������������ĿĻĳĭĦģħĩİĳľĿ�@�L�Y�[�a�]�Y�L�H�@�>�:�@�@�@�@�@�@�@�@�H�T�a�k�m�t�v�m�a�T�J�H�;�/�;�B�H�H�H�H�����������������������������������������������ûɻ˻û������x�l�S�P�P�S�_�l�x��������������ܹҹϹù��ùϹܹݹ��D�EEEE*E4E*EEED�D�D�D�D�D�D�D�D�D��������"���	����������������������Z�f�s�����������������s�f�[�Z�S�S�Z�Z�������������������������������������ѿ.�;�G�T�\�j�x�u�w�m�T�G�;�7�*�����.���	��"�.�8�0�.�"��	������������������ŇŠ�����������������ŹŪśŉ�łŇ��������ÿ��������)�B�a�j�k�a�B�)�����'�4�@�Y�e�r�u�r�f�Y�M�4�'�������'āčĚĦĨĦĦĚĘčā�}�~�zāāāāāā�.�E�F�>�$���ʾ��s�Z�K�N�f������Ծ��.�����ʼ׼߼ּѼʼ��������������������������(�4�A�O�Z�f�s�}�s�f�Z�M�4�����������������������������������}�|�����������m���������x�m�a�T�H�E�>�2�,�,�5�B�H�T�m�Z�\�e�b�Z�M�K�I�M�W�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�s�������������������s�n�k�g�o�s�s�s�s���"�$�#�"���	���	�	�����������������������������������z�����������ſ����������������������������������������y���������y�m�j�m�q�y�y�y�y�y�y�y�y�y�y�ݿ������ݿӿѿпϿѿӿؿݿݿݿݿݿ���!�(�.�(�%�*�5�4��������������f�s�}�������������s�f�Z�T�Z�]�\�]�e�f�(�4�A�K�B�A�4�1�(�����#�(�(�(�(�(�(�����������Ƽ�������������r�m�f�c�r���àáìöìäàÓËËÓßàààààààà�������������߼ۼ�����������������&�(�3�(���������������#�/�<�H�U�X�V�U�H�<�/��
��������
�àìù��������������ìàÓ�z�o�q�zÅÚàƁƎƚƧƳ����ƳƨƦƚƎƇƁ�z�s�r�u�xƁ�����ĽͽĽ������������������������������r����������ļȼü�����r�Y�@�6�;�A�L�r�л����'�4�<�<�?�6�'����ܻɻû������й����������	���ܹù������������ùϹ�EiErEuE|EuEnEiEdE\E[E\EgEiEiEiEiEiEiEiEi���������������y�l�`�W�S�U�R�H�G�`�l��������������������������������������������EuEzE�E�E�E�E�E�E�E�E�E�E�EuEtEuEwEuEqEu�����!�-�:�@�S�[�N�F�-�!����������uƁƅƎƚƧư������ƳƧƎ�u�h�^�e�h�s�u������������������������������������������������������������������������������ػ:�F�_�x�����������l�S�F�:�-�!���!�-�:D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��0�6�<�I�S�W�U�Q�I�A�<�0�%�#����#�%�0�~���������ֺ����ֺɺ��������y�r�v�~���	�������	�������������������������#�-�/�:�/�#� ������������6�C�O�T�O�O�C�7�6�6�.�4�6�6�6�6�6�6�6�6 N L 7 7 % p t / ? J C C K  B 2 n T s Z * W 4 M B S M l B : K ' K : ] - P = v ' L B 4 M O ? N e t  � B  ^ > � G    c  �      �    �  �  ,  �  �  �  �  �    �  M  b  �  �  �  `    U  �  3  <  �  �  g  z    e  
  R  .  *  %  �  �    &  -  s  Z  k  �  �  �  �  W  $  g  �  p  t  Z��h�ě�;ě�;�`B=49X<�t�<�j<���=o=L��=}�<ě�=�9X>�$�=@�=C�=��=�P=<j='�=���=+=��=+=P�`=��=\)=�w=8Q�=T��=8Q�=���=]/=49X=]/=���=��=�O�=Y�>C�=>o=u=���=y�#=�-=��-=�O�=�%=��
=��><j=���=��=���=��=��B
tB RA�r�B&I�B"�jB�(B3B!�B��B�NB�HB�YB�B	Z-B"XB �&B�vB"��B#^�B�_B�B֢B��B �wB��B+yB�aB�]B��B�!B��B�@B"c>B%i�B�B�DBJB�B -B�Bq@B�
B��B,�'B"8SBw�B�B��B��B0=B��B�'BQzB��BڢBоB��B�XBAZA�{bB&F]B"��B�}B@WB?ZB�{B5�B��B*�B�FB	BLB!�B ��B�B"�6B"�`B��BDdB 7BFQB Q�BB.BA%B�BGqB��B��B B"?�B%A*B7
B�BI�BC�B��B��BW�B�@B�IB,��B"@
Bc�B��B?�B��B�B�RB1�B+�B@�B@GB@BP�A�6b?�A+A�%�@�-.@�?�?;?C�h.A���ACwA�ںAeA]4�A�A��b@���AޘTAQ*@���A8��A���A�A>�AEAA\�6A���Art�Am�JA|��A��rAB��A8z6@�S�A�M�Az�A��NA���A�"B0tA$S�@咿@�X�>�>�C��Ad A!@{C�8@jD~B�1B��A�Y@�y	C���A��5@'�A�q�A��/B �dA�Z?�*�A�pa@��@�f?Mc�C�g�A�ݕAD�A�u�AckA^,�A��OAԍS@ː�A���AQ��@���A8�xA���A��lA?,AE�5A\��A�rAAr?�Anz A}A��lAC�)A8�2@��A˕A�A��A�p-A�j�B@sA%?@�%@�&�>�ǉC��%A�KA!n5C�D@d�Bh�BLKA�#@x�3C��BA��@fA��OA�+B �5   
            0               #   .      F  o         4            5                                 %            !         
   \   I   R      !   
                      w                	               #               '   #      )   2         ?            !                                                         '   #   +      %                              !                                       %         #            =                                                                           )                                             O�N��LN��rNơN��NQ��N2gN���N��+Oٞ�O���N�2�O��O̖�Oj_�N���Pl�O�Oe�lO9͈O=	�N�'N��,N"��O���N�W$N4jNL�SON��O�#Nt_kO)�3NL�lM�$qOA�OK�4O\8O{)2NG�_O�0�Ol\EP�N	/DO2ҙN0��O	D�Om8;OBB�NQ6<N���O]��O"�&O*S�OEO*ZeN"E�NEV�  Y  �  �  G  P  <  `  )  N  �  �  �  	     �  �  �  1  a  �  A    @  ^  �  �  c  _  L  {  �  �  (  �  /  �  �  =  P    	r  	�      u  	=  �  ;  �  �  -  B  ?  �  �  �  *��㼃o�D�����
<���<t�<49X<�o<��
<�t�<�`B<�C�=��>?|�<�9X<�9X<ě�<ě�<���<�/=<j<�h<��<��=o=o=o=\)=\)=#�
=#�
=L��='�='�=,1=L��=@�=8Q�=8Q�=���=�C�=]/=H�9=q��=T��=m�h=e`B=ix�=ix�=�+=��=���=�C�=���=�{=�Q�=�v�rtw��������������|tr\\bht����vth\\\\\\\\"/04/-$"&)/0<?IJQSRLI<40&&&&����������������������������������������E?>@EHKQQNJHEEEEEEEE:52:<HLOUWWUH<::::::����������������������5BNX\f[5)���+).8B[gt�����tg[NH7+��������	
���������www���������������zw@@CN[gt�������tg[NC@��������������������{z}�����������{{{{{{��cr{zmjnh[NL=1���������������������������������������������������������5:BGN[gnttslg[NMB;65'�����������������������������������������������������������������������������������

���������������������������������
#/<DIHD</# ���������

�����##./2/+# NX[cht������xtnhcZSN��������������������"#(0120#%)/59BHFFEB@75)��������������������������������������������������������������������������������������"*,,) ��0/-,/6BO[^_^^^]ZOB60_gnz�������������sb_-/1<AHLH=<:/--------����������������������������������������������
"
������������ %)��
)412220)31356BNQNNB>75333333)+.,)yyvsnz������������|y��������

�����
!)15=>52)&)4BDLRPH@;6)����!''!�������������������������##,///#Ŀ��������������ĿĻĳĭĦģħĩİĳľĿ�@�L�Y�[�a�]�Y�L�H�@�>�:�@�@�@�@�@�@�@�@�T�a�g�m�q�o�m�a�W�T�H�E�H�L�T�T�T�T�T�T�������������������������������������������������������������������z�x�w�x�z���������������������������������D�EEEE*E4E*EEED�D�D�D�D�D�D�D�D�D���������������������������������f�s�x�����������s�f�d�]�^�f�f�f�f�f�f�������������������������������������޿;�G�T�`�f�j�n�k�`�T�G�;�/�'�!���"�.�;���	��"�.�8�0�.�"��	������������������ŠŹ�������������������ŹŧŞŖœŘŠ���)�=�H�J�G�?�6�)�����������������'�4�@�M�Y�c�r�r�Y�M�@�4�'�������'āčĚĦĨĦĦĚĘčā�}�~�zāāāāāā�.�B�C�.�����ʾ��s�Z�N�Q�f�s�������׿.�����ʼ׼߼ּѼʼ��������������������������(�4�A�O�Z�f�s�}�s�f�Z�M�4�����������������������������������}�|�����������a�m�u�z�}�y�r�m�b�a�T�H�C�;�:�;�B�H�T�a�Z�\�e�b�Z�M�K�I�M�W�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�s�������������������s�n�k�g�o�s�s�s�s���"�$�#�"���	���	�	�����������������������������������z�����������ſ����������������������������������������y���������y�m�j�m�q�y�y�y�y�y�y�y�y�y�y�ݿ������ݿӿѿпϿѿӿؿݿݿݿݿݿ���!�(�.�(�%�*�5�4��������������f�s�}�������������s�f�Z�T�Z�]�\�]�e�f�(�4�A�K�B�A�4�1�(�����#�(�(�(�(�(�(����������������������{�r�p�r�v�������àáìöìäàÓËËÓßàààààààà�������������߼ۼ�����������������&�(�3�(��������������
��#�/�<�H�P�R�M�H�<�/�#���
�����
ìù��������������ìàÓÇ�z�u�{ÇÓàìƁƎƚƧƲƳƾƿƳƧƚƎƈƁ�z�s�s�u�zƁ�����Ľ̽Ľ������������������������������r�������������������r�f�Y�M�J�I�N�Y�r�лܻ�����&�-�-�'�������ܻ̻ûŻй���������������ܹù����������ùܹ�EiErEuE|EuEnEiEdE\E[E\EgEiEiEiEiEiEiEiEi�������������������y�l�`�Y�[�Y�`�l�y��������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�EvEuEuEuExEwE�E������!�-�:�@�S�[�N�F�-�!����������uƁƅƎƚƧư������ƳƧƎ�u�h�^�e�h�s�u�������������������������������������������������������������������������������ػ:�F�_�x�����������l�S�F�:�-�!���!�-�:D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��0�6�<�I�S�W�U�Q�I�A�<�0�%�#����#�%�0�~�������������ɺֺֺɺ����������}�v�|�~���	�������	�������������������������#�-�/�:�/�#� ������������6�C�O�T�O�O�C�7�6�6�.�4�6�6�6�6�6�6�6�6 N L - #  0 t 4 ; I > C H  ? 2 o T s Z  W 4 M B S M l B : K & K : ] . F @ b * M B 4 < O - N e t  � 0  R > � G    c  �  �  �    ^  �  �  �    �  �  �  �  �  �    b  �  �  �  `    U  �  3  <  �  �  g  z  q  e  
  R  �  �  �  R  x  �  �  -  �  Z  )  �  �  �  �  W  \  g  �  p  t  Z  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  Y  K  <  *      �  �  �  �  �  �  n  X  @  (    �  �  �  �  �  �  �  �  �  �  �  �  {  f  Q  =  ,      D  �  ~  k  t  �  �  �  �  �  �  �  �  �  �  �  �  g  #  �  }    �  S  4  A  C  ;  1  &    
  �  �  �  �  �  �  J    �  �  _     >  X  q  �  �  �  
  *  B  P  J  B  6  &    �  S  x  �  �  �  �  
  -  9  3  '      �  �  �  �  ]  5    �  �  \    `  I  0    �  �  �  �  n  D    �  �  w  7  �  �  k  $   �     %  )  )  '      	  �  �  �  �  �  �  �  n  ^  N  7    5  <  D  K  N  N  M  I  D  <  0    �  �  �  u  <    �  }  x  �  �  �  �  �  �  s  ^  H  %  �  �  �  q  +  �  �  p  �  ^  �  �  �  �  �  �  �  �  l  F    �  �  l    �  a  �  5  �  �  �  �    {  w  s  n  i  d  ^  X  R  M  I  E  L  W  a  �  w  �  �  	  	  	  �  �  �  j  >    �  �    N  u  �  �  �     �  h  �  �  V  �      �  x  �  �  h  b  �  =  �  R  �  �  �  �  t  f  W  F  2    �  �  �  s  7  �  �  ^    d  �  �  z  q  h  ]  N  8      �  �  �  �  u  i  b  ^  a  h  �  �  �  �  �  �  �  �  �  �  �  d  G  2    �  �  1  �    1  -  '           �  �  �  �  �  �  �  n  9  �  �  �  Q  a  S  A  :  F  2            �  �  �  }  @  �  �  M  �  �  �  �  �  }  o  X  ?  /  E  T  S  K  -    �  �  r  3  �  �  H  �  �    '  <  A  -    �  �  c    �    H  q  s  �    �  �  �  �  �  �  �  �  �  �  �  �  |  m  ]  L  ;  *    @  =  ;  8  4  0  )  !        �  �  �  �  �  �  q  I  "  ^  O  @  1  "       �   �   �   �   �   �   �   �   �   �   �   �   �  �    n  S  6    �  �  �  �  �  z  [  2    �  �  ~  r  �  �  �  �  �  �  {  u  m  d  [  P  C  5  %    �  �  �  �  �  c  V  I  <  /  #      �  �  �  �  �  �  w  D     �   �   y  _  Z  U  Q  L  E  8  +        �  �  �  �  �  �  �  p  ]  L  =  -      �  �  �  �  v  V  6  %        �  �  �  �  {  {  x  r  g  Z  H  6  #    �  �  �  �  �  c  ;    �  �  �  �  �  �  �  v  ^  E  0      �  �  �  �  �  �  �  �  �  @  l  ~  �  �  �  �  �  �  �  k  ?    �  L  �  b  �  &  p  (    �  �  �  �  s  T  .    �  �  �  G    �  �  M  �  �  �  �  �  �  �  �  �  �  �  �  }  n  ^  P  E  ;  0  &      /  (  !            �  �  �  c  3    �  �  i  9  �  `    Q  q  �  �  �  x  d  J  (  �  �  i  U  ;  �  k  �  (  �  t  �  �  �  l  U  ;    �  �  �  }  R    �  �  B  �  f  �  /  9  (    �  �  �  p  Y  G  )  �  �  q  ,  �  I  �  &   R  (  ;  N  F  ;  (    �  �  �  �  �  O    �  �  g  '   �   �  	�  
3  
�  
�  
�  	      
�  
�  
q  
!  	�  	O  �    7    �  V  U  �  �  	,  	U  	h  	r  	i  	`  	R  	.  	  �  �    �    i  �    	�  	�  	�  	�  	�  	|  	z  	e  	U  	6  	   �  M  �  8  }  �  �  �  <    �  �  ~  L    �  �  �  L    �  �  G    �  y  4   �   �  �  �  �  �  �    �  �  �  �  �  �  a  .  �  �  }  ;  �  n  u  ^  F  &    �  �  �  z  R  (  �  �  �  }  N    �  �  P  �  	  	!  �  �  �  g  /  �  �  ]    �  H  �  j  �  c  �  �  �  �  �  ~  `  :  
  �  �  X    �  i    �  &  �  y     �  ;  0  $      �  �  �  �  �  �  d  ?    �  �  @  �  �  }  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  Z  B  (    �  �  �  y  N  !  �  �  /  w  -       �  �  �  �  n  8    �  �  Q    �  N  #  M  k  C  �  �    &  6  A  >  "  �  �  ~    j  �  �  �  O  �  c  /  ?  8  /  !      �  �  �  �  �  �  �  w  a  =        	  �  �  �  �  �  �  �  ]  0  �  �  �  g  b  *  �  t  �  �  	  �  �  }  p  c  Q  ;  !    �  �  �  �  p  T  (  �  _     �  �  �  }  c  C    �  �  �  �  U  '  �  �  �  g  5    �  �  *    �  �  �  �  �  z  b  K  8  (      �  �  �  �  �  �