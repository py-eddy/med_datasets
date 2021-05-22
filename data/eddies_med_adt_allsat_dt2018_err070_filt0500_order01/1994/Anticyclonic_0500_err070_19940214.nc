CDF       
      obs    0   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�j~��"�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       Pk��      �  l   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       >o      �  ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E�Q��     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�z    max       @v�z�G�     �  'l   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q@           `  .�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  /L   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >E��      �  0   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,0      �  0�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,?�      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Q�   max       C��      �  2L   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Q]%   max       C��      �  3   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          p      �  3�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  4�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  5L   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P2��      �  6   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?�|����?      �  6�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       >o      �  7�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E��\(��     �  8L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�z    max       @v�\(�     �  ?�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q@           `  GL   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  G�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  Hl   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?�|����?        I,               2   	      !   
   o   G   
                  7   "                  "   2                  *   
            
   
                   e            O��1NJ��N�\N�Pk��O ��N��@O�4�O1˅P�kPd�jNg��O�!�N �N�E}N�ϞO�@(Oǫ�P$�M��OhO5�EN��NO��P�>P��OO�N���OL.�N��O7�O��O��NȤ�O��Np�NNu�rO�{	N%��O\N��ODdO�e�O�Oj/�NKMNk'��`B�T���#�
�ě�;�o;��
;ě�;�`B<49X<T��<T��<T��<���<�1<�1<�9X<ě�<ě�<ě�<���<���<�h<�h<�h<�h=o=+=\)=�P=�w='�=49X=49X=49X=<j=<j=@�=H�9=L��=T��=�o=�\)=�hs=�v�=ȴ9=���=�`B>oV[^\\gkt��������ug[V60:<HTQHD<6666666666y�������������yyyyyyrpt�����trrrrrrrrrr&2<Ujo�������zaU<%@@BMO[hmtwtlklh[OIB@�������������������
#/>CMSH</#
����������������������b\bz���������������b������������������������������������������������������������������������������-,($-0<>HB=<30------��������������������	)5;?B>85)	#/<H_eZTXUH<#('(.0;BHUnz����zU7/()),1558::95)))))))))��������������������RNQW[cgntx����}wkg[R������ ����	)BO[honh[OB6���������������������������)--$����ntw�������������|xtn�����������������������������������������
#/0650,#
����><>BIN[fjpqohf[UNJB>�������!%"���������������� ������� �	!)+*)  ���)BNTTPHB5)������������������������!).,) ����]\\ainszznga]]]]]]]]��������������������26:;ABDOYSOB96222222LMNOT[\dhtzxtn_[OLjmoz{�����������zmjj-6;<64)����)-��������
�����`agmqz������zmda^^``kcbmz�����������zsmk"/10/&"���������������������#�,�H�a�zÏÓ×ÏÂ�z�n�a�Z�P�/�#���#���	�����	�������������������������������������������������������������������L�Y�b�e�o�e�Y�L�J�J�L�L�L�L�L�L�L�L�L�L�m�����������������������v�w�r�M�D�F�T�m�������ĽƽĽƽĽ���������������}�������Z�f�s�w�����������s�f�`�Z�O�T�Z�Z�Z�Z�`�m�������������y�m�`�G�>�<�8�6�;�B�[�`�ܹ�����'�.�'��	�����������ܹ۹ڹ��B�O�[�h�t�r�m�[�O�B�)���������!�B����B�[�g�x�t�[�)�����������������ÓàâããàÓÇÀÃÇÍÓÓÓÓÓÓÓÓFFF$FVFsF�F�F~FoFVFJF6F:F.FE�E�E�E�F����������������������������������f�r���������������r�p�f�c�f�f�f�f�f�f��	���������������������)�5�B�N�[�g�t�y�}�x�t�[�N�5�)�&����)�������������������ùôîì�������5�A�Z�g�s���������������i�W�L�K�B�5�.�5�����������������������������������������	���"�$�,�.�(�"��	��������������	�	������$�5�;�A�M�A�5�(������������"�.�;�?�;�7�.�"���	��������	����ʾ׾�������������׾ʾ������þƾžƾʽ��н���������н������������������T�m�����������������`�T�G�;�+�-�5�6�G�T�<�C�H�Q�U�V�U�P�A�<�/�#���� �#�/�7�<āčĚĦħĳĳĳİĦĚčĉā�Āāāāā�y�������������������������y�q�l�b�l�q�y�������������ڼּҼԼּ����������������������������������������������������������������������������������������"�/�;�;�@�?�;�/�*�"��	��	�	��������������� �����������������������������#�0�<�K�R�V�T�N�G�<�#��
���������
�#��#�#�/�4�<�H�<�/�#�"����������M�Z�d�f�h�g�f�a�Z�M�D�A�@�A�D�H�M�M�M�M�����ûɻû������������������������������s��������������������j�f�V�N�O�Z�b�s���������ɺ̺ɺ����������������������������'�(�4�?�@�M�V�Y�M�@�4�'� ������T�]�a�k�m�z�~�����z�m�i�a�Z�T�R�N�O�T�T�3�)�'�������������'�3�7�@�A�@�3D{D�D�D�D�D�D�D�D�D�D�D�D{DwDoDjDiDjDoD{������������������ŹŭŭţŤŭųŹ������àù������������������ùìãØÎÏÓÛà�
��� ���
����������
�
�
�
�
�
�
�
�ּ�������������ּҼѼּּּּּ� U D f G R , N ? I P = B h n J : 5 % > � 2 T e g 3 2 1 , N ? U = G ) - � B \ ( j ` g .   . . U  �  c  �  2    d  �  �  �  �  �  �  <  v  �  �  !  �  �  e  h  �  �  z  q  �  R  �  �    �     &  �  �  �  �  w  -  �  U  +  �  �  '  �  W  ��o�ě���o:�o=Y�<T��<49X=#�
<��
=��=���<�j=8Q�<���<���<�`B=<j=��w=ix�<�/=#�
='�=+=49X=�%=��=aG�=D��=T��=L��=]/=� �=Y�=P�`=��P=�O�=ix�=q��=��=m�h=�E�=�-=��>E��=�>   =��>hsB	ƋB��Bg�B�B�B�BH�B��B!qB{B�B"�B��B!�B%��B��B�LB��B�>B�BxQB	R�B�B?B!iB��B�-B,�B,0B$�IBy�B�GB}�Bg1B�B�B�B�MBܱB<FB�[B <B�>B��A��B @�A��B	B	�\BCB��B��B?�B�WB@6B@�B ��BC�B�?B"?�B�B!��B%��B��B�B�jB��BBfB��B	?�B��B�B!��B��B�WB?�B,?�B%=;BDB@�BO�B@�BE�B?�B��B�OBC�BB<B�B ��BIJB��A�sEB BqA��B?rA�0?A��X@�+?�uA���A!>ABamAj\?Q�A�R�A��A�b�C��@���@�Y�A�>�A��Aђ#A�n5B�^A���A�hfA^1ASplA)��Aj�A�HA�\MA4�A�!A�:�A���A�}A�SEA���A�
NA>3~@���AD�@%|�@�>�A��?��C�ÑA�1cA�*?A�rAaA�{A���@� �?���A�#A!�@AB�AjMy?Q]%A�~A��hA�fC��@��@��XA�~�A�p�A�T�A�ctB�!A��A��A]~�AR�A-	�Aj��A���A߁�A��AW�A�k�A��fA�>�A���A�EA0A? �@� �AC�@+��@��gA��.?�:0C���A�u�A�ԴA�/Af1               2   	      "   
   p   G                     7   "                  #   3                  +   
                              !   e               !            9               '   ;      )               #   '                  '   )                  %         !                                                      /                  -      !                                    '                              !                                       O��NJ��N�\N�P2��N٩	N��@O�G�O1˅O#��P'�N�tO�tYN �N�E}N�ϞO&O]`�Oc! M��OhO5�EN��NO\�P��Os�N���N���N��N��N�շO��!O��NȤ�O��Np�N�i2Nu�rO��N%��N��N��ODdOI��O�Oj/�NKMNk'  �  ~  !  �    �  �  s  �  �  �  [  /  f  �  �  �  �  �  �  �    �  3  �  �  
  	  �  >  G  �  �  [    �  L  �  E  �  �  g  f  c      O  ��o�T���#�
�ě�<49X;�`B;ě�<T��<49X=�O�<ě�<u<���<�1<�1<�9X=o='�=\)<���<���<�h<�h<��<�=Y�='�=\)=49X=�w=<j=T��=49X=49X=<j=<j=H�9=H�9=P�`=T��=�C�=�\)=�hs=�;d=ȴ9=���=�`B>ogggefghtt�������{tgg60:<HTQHD<6666666666y�������������yyyyyyrpt�����trrrrrrrrrr"*<Uaz�������zkU<$BABFO[afghihd[OKGBBB�������������������
/9?HKJH/#
����������������������������������������������������		�������������������������������������������������������������������-,($-0<>HB=<30------��������������������)568:;850) #(/<AHRWTMHF</ 7399:>HUbmtrnmaXUH<7)),1558::95)))))))))��������������������RNQW[cgntx����}wkg[R������ ����!)O[hmlh[OB6)��������������������������������������������������������������������������������������������
#/0650,#
����A@BNZ[bghgb[NNGBAAAA������! ���������������� ������� �	!)+*)  ���)BNTTPHB5)�����������������������),)(������]\\ainszznga]]]]]]]]��������������������26:;ABDOYSOB96222222PQW[_fhtx}}vtjh[PPPPjmoz{�����������zmjj-6;<64)����)-���������
�����`agmqz������zmda^^``kcbmz�����������zsmk"/10/&"���������������������<�H�U�a�n�u�zÇÇÀ�z�n�a�U�H�D�<�0�8�<���	�����	�������������������������������������������������������������������L�Y�b�e�o�e�Y�L�J�J�L�L�L�L�L�L�L�L�L�L�m�������������������������~�{�a�K�M�a�m�����������������������������������������Z�f�s�w�����������s�f�`�Z�O�T�Z�Z�Z�Z�m�y�����������y�m�`�T�G�F�>�=�B�M�T�`�m�ܹ�����'�.�'��	�����������ܹ۹ڹ��6�B�O�[�[�_�]�[�R�O�B�6�)�&�$�%�)�-�6�6��5�B�W�g�t�x�r�f�N�)�������������ÇÓààáàÓÇÂÅÇÇÇÇÇÇÇÇÇÇFFF$F1FJFjFyF~F|FoFcFVFJF=F5F$FFE�F����������������������������������f�r���������������r�p�f�c�f�f�f�f�f�f��	���������������������B�N�Q�[�g�h�l�h�g�[�N�B�5�0�)�%�$�)�5�B��������������������������ÿ�����N�Z�g�s���������������s�j�g�X�N�L�J�J�N�����������������������������������������	���"�$�,�.�(�"��	��������������	�	������$�5�;�A�M�A�5�(������������"�.�;�?�;�7�.�"���	��������	����׾���������׾ʾþ����žȾɾȾʾӾ׽��н����������н����������������`�m�y���������������y�m�`�T�G�D�E�I�T�`�<�=�H�N�I�H�<�9�/�)�#�!�!�#�/�3�<�<�<�<āčĚĦħĳĳĳİĦĚčĉā�Āāāāā�����������������|�y�p�u�y���������������������������ڼּҼԼּ����������������������������������������������������������������������������������������"�/�;�;�@�?�;�/�*�"��	��	�	��������������� �����������������������������#�0�<�K�R�V�T�N�G�<�#��
���������
�#��#�#�/�4�<�H�<�/�#�"����������Z�_�f�f�f�f�^�Z�N�M�C�G�M�N�Z�Z�Z�Z�Z�Z�����ûɻû������������������������������s���������������������m�f�Y�P�R�Z�f�s���������ɺ̺ɺ���������������������������'�4�;�@�M�Q�U�M�@�4�'�$��������T�]�a�k�m�z�~�����z�m�i�a�Z�T�R�N�O�T�T�3�)�'�������������'�3�7�@�A�@�3D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DpDnDpDxD{������������������ŹŭŭţŤŭųŹ������àù������������������ùìãØÎÏÓÛà�
��� ���
����������
�
�
�
�
�
�
�
�ּ�������������ּҼѼּּּּּ� \ D f G N / N C I - ( 0 q n J : 2  5 � 2 T e g 2 ' ' , X ? K 0 G ) - � L \ # j W g .   . . U  O  c  �  2  ^  �  �  K  �  [  �  '  �  v  �  �  Q  �  �  e  h  �  �  5  j  �  �  �  �    �  n  &  �  �  �  �  w    �    +  �  �  '  �  W  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  8  G  Q  \  j  x  �  �  �  �  }  S  $  �  �  �  H  �  {  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  w  z  }  !          �  �  �  �  �  �  �  �  w  c  N  :  (      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  l    �  �  z    9  *  �  �  �  �  �  �  �  �  �  �  s  ^  H  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  _  F  -     �   �   �   �  ?  \  k  r  r  l  \  E  &    �  �    B  �  �  [    �  *  �  �  �  {  t  m  g  d  b  ]  U  J  9  -  -  #  �  �  �  �  	7  
&  
�  M  �  �  �  &  g  �  �  �  [  �  �  
�  	�  �  �  (     �  �  �  �  �  �  |  ]  5    �  �  =  �  X  �  �  U  �  H  J  M  T  Y  S  I  5      �  �  �  |  V  0    �  �  �  �  �    "  -  -      �  �  �  o  @  
  �  t    �  �  �  f  d  b  `  Z  I  7  %    �  �  �  �  �  �  g  L  0     �  �  �  �  �  �  �  �  �  �  �    x  r  l  f  _  Y  S  L  F  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  j  `  U  J  q  �  �  �  �  �  �  �  �  �  b  =    �  �  �  J    	  5  �  �  E  y  �  �  �  �  o  N  0    �  �  [  �  H  �  �  I  |  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  7  �  [  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  |  f  N  7      �  �  �  �  g  Q  <  !  �        �  �  �  �  �  �  �  v  \  >  $    �  �  �  �  \  �  �  �  �  �  �  �  �  �  �  }  m  \  L  ;  *       �   �  &  /  2  1  +      �  �  �  �  �  �  �  k  ?  �  �  -   �  �  �  �  �  �  �  �  �  �  �  s  Y  3  �  �  \    �  �  �  b  d  l  q  w  ~  �  �  �  �  �  t  O    �  e  �       '  X  �  �  �  �  �    	    �  �  �  y  J    �  �  {  H  �  	  �  �  �  �  �  �  �  �  �  k  S  =  ,    �  �  �  z  Q  W  a  l  y  �  �  �  �  �  �  �  �  �  �  �  l  C    �   �  >  3  (      �  �  �  �  �  |  b  H  1      �  �  �  �      "  *  2  :  B  F  F  A  2      �  �  {  +  �  s    �  �  u    �  �  z  `  0  �  �  c    �  {    _  �  �    �  �  �  �  �  �  x  |  �  �  }  q  a  O  ;  %  	  �  �  3  [  O  B  5  '    	  �  �  �  �  �  �  |  i  Y  J  T  h  }    �  �  �  �  �  �  �  d  @    �  �  �  [    �  i  /  �  �  �  �  �  �  �  )  	  �  �  e  #  �  �  N    �  _  	  �  4  ?  J  K  J  A  5  &    �  �  �  �  u  N  &  �  �  Y  !  �  �  �  �  �  r  a  T  F  7  &          �  �  �  `  3  9  B  D  @  8  -       �  �  �  �  v  F    �  s     �   d  �  �  �  �  �  �  �  {  `  F  )  
  �  �  �  �  �  d  G  +  �  �  �  �  �  �  �  a  :    �  �  d  '  �  u    �  �  �  g  K  (        �  �  �  �  `    �  w  7  �  �  =  �  q  f  B  3  !    �  �  �  �  �  u  &  �  ]  �  �  %  �  /  �    9  T  c  R  /     �  o    �     E  y  �  `  	�  p  �  7    �  �  �  h  >    �  �  �  P    �  �  ^  !  �  �  �       �  �  �  �  �  |  l  T  5    �  �  r     �  N  �  �  i  O  7    �  �  �  j  6    �  �  Z     �  �  �  Y  5    �      �  �  �  �  �  �  t  Q  +    �  �  �  V  '  �  �  �