CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�C��$�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�1   max       P�h�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       >J      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E���R     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vn=p��
     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @M            l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @��@          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >Q�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�A�   max       B,�M      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?c>   max       C�I�      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P�h   max       C�W�      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�1   max       PC��      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���+j��   max       ?�%F
�L0      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >C�      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E���R     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vn=p��
     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @M            l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @�<�          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?vOv_خ   max       ?���[     �  N�         �   ;   (   $   	            (            U               X   +   
   +      	   
            1               '                     	                  
   $   J            1N�\�O��P�h�O�;WOt�JO�lKOf��N���N&�N��sO��PN���OZS�PR�P[�,O�dN�6�N�<�O�9P=�5P5�iN�$P��M�1Oa{O"�NC7dN-ׄO��P�8ON�$:N�D=O�9vN�AP@�Ng#+Ohm�OI��OJ�N���O//�N�TO	3gN1��N�:�O�HN�zNo�nO`�zO��N&�YN��N��O]4���ͼ�C��o���
��o;o;�o;��
<#�
<#�
<49X<D��<T��<�t�<��
<��
<�1<�1<�1<�9X<�j<�j<���<���<�/<�<�<�<��=o=o=+=\)=t�=�P=�w=49X=8Q�=8Q�=<j=@�=@�=H�9=L��=P�`=T��=�%=�t�=���=� �=��#=��m>   >J��������������������b__`^gt����������tgb()5�������}d[B7)����������	������okijlrt�����������to/;HKV`gg[TH;"nmpt��������������tn����������������������������������������������������������������������������������������������������*.?BO[ehlhmsh[OB>66*O^gt������������of`O�����)GNPJA5)���HSUbjn{���{undb[UIIH)5785)EOPX[chmqrskhe[OEEEExvvu��������������xx�����������������5N[mt���t[NB�'$"&)1159BEJJIB;5)''rvuz|�������������zr����������������kjmvz�����������zmkk~~�����������������~)..)$).6@BDB<6+)$$$$$$$$����������������������5BHNSd_U5)����������������./3<BHPUaUKH<80/....)6BN[fkoog]YB5)}���������������}}}}��)6BOXXUOB6������#(-/1/+#"�������������������������),,*)#������������������������������������������-+,./3=@HLTPTROH;5/-DCBHKT]abaaXTHDDDDDD����������������������������������������##$09<A<0+&# � 
!#$''&*$#
`^fhtvvtkh``````````
!#%%#
��������������������
	���������������������������������������������������	


��������yz����������������zy�������������������������~�}�~���������������������������������������������������O�hāĦįĭĒą�t�[�6�)���������1�O������<�M�Y�g�Y�P�4�����ۻջԻܻ���zÇÓàì��������ùìáàÓÇ�z�n�m�p�z�/�H�T�a�k�{�|�s�m�a�T�H�;�"���	���/����#�2�A�D�H�G�A�<�4�,�(�"������������Ŀѿݿ׿ѿĿ��������������������������������������������������������������ÓØàéìïòìâàÓÇÄ�~ÁÅÇËÓÓ��(�5�Z�s�z�����������s�g�Z�N�0������ �(�,�,�(����������������ѿݿ����ݿؿѿĿ������������������Ѿ����������������~�f�Z�A�,�(�)�A�M�s�������F�`�n�n�b�X�<�0����������������弤���������¼���������������~����������Z�f�i�i�j�h�f�Z�W�N�P�S�Z�Z�Z�Z�Z�Z�Z�Z�H�S�U�a�f�n�x�n�a�U�H�<�:�<�@�?�H�H�H�H�#�/�<�H�M�T�N�H�A�<�/�/�/�#����!�#�#ìî������������������ì�z�c�S�K�O�zØì�	�H�T�]�_�X�W�S�H�:�/�������������	�G�T�`�m�q�y��������y�m�`�T�K�G�E�F�G�GŔŭ������������������ŹŭŎŌŎŒŖœŔ�r�s�u�s�r�f�c�\�f�q�r�r�r�r�r�r�r�r�r�r�ĿѿԿݿ���ݿۿѿĿ������������¿Ŀ������������������������s�g�Z�Z�g�s�������a�g�k�m�a�U�I�M�U�W�a�a�a�a�a�a�a�a�a�aÇÎÓØÓÐÇÃ�z�q�z�}ÇÇÇÇÇÇÇÇ�G�`�q�r�m�c�^�T�.���	�����"�.�;�GƧƳ����0�K�=�Z�K�$���ƧƎƁ�c�]�cƁƧ��������������ܻۻܻ������������
����
���������������������������!�1�5�:�@�>�A�A�5�(�$����Կֿ�����������������������������������������׾׾���������׾������������������ʾ��"�/�;�>�;�7�/�"����	��	���"�"�"�"�����������½����������������������������Y�e�r�~���������������~�r�e�[�Q�L�E�H�Y�����'�/�3�3�'� ��������������m�z���������������������z�u�m�l�m�m�m�m�����
���#�0�<�A�<�0�#����������������U�b�n�y�{�|�{�w�n�b�a�U�J�P�U�U�U�U�U�U������������������������������������}��M�M�W�M�A�4�(�$�(�4�A�M�M�M�M�M�M�M�M�M�s�������������������{�~�s�f�a�f�h�s�s�׾����	��������׾ʾ����������ʾξ׼'�4�@�K�L�@�4�0�'�%�'�'�'�'�'�'�'�'�'�'ǡǭǵǹǭǫǡǔǏǎǔǝǡǡǡǡǡǡǡǡ�����������
����
�������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D�D�D�E*E7ECEPEVEPEIECE7E4E*E*E*E*E*E*E*E*E*E*�x�������������û����������x�s�x�x�x�x�xE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ@�M�Y�d�f�q�f�Y�M�@�4�'�������7�@  # 7 9 K  S 8 t @ G > 0 < % ; m L : ; A = 8 Z C p D 8 W e > > X ] B g a I : U _ 6 V U Q g L C / P E t H A  �  0  �  Y  �     *  �  `  &  �  �  �  �  �  D  �  �  ^  ]  B  %  �  8  <  �  o  N  �  �  �  �  �  �  w  �  &  �  �  �  �  �  Y  j  �  n  G  k  �  #  V  �  �  �ě�:�o>Q�=Y�=t�=��<T��<D��<D��<��=L��<�o<�=+=���<��<���<�h=,1=�G�=��=o=�C�<��=\)=#�
=t�=t�=8Q�=��T=8Q�=<j=��=#�
=���=8Q�=]/=�o=}�=Y�=�+=e`B=���=m�h=e`B=��=�O�=��=�`B>"��>I�>
=q>�>49XB"C8B
*B	!oB"g�B
?hA�A�B��B�SB �B!��B�nB[�B��B
�SB�B'�3BG]BJ�B�+B�<BJgBRB�EB9B UrB
BȐB��B!��B��B��B��B'|B&B��B�KB+ʧBB�B!G,BjA�LA�FB��B6�B%q/BC8B��BB�B�B�pBLB,�MB��Bh�B"@CB	�JB	?>B">�B
?�A���B��B*�B!6�B"@�B6;Bn�BH�B*B��B'�B��B{FB�B�,B)�BBB�BC�B ��BA0B�B�nB!�~B=pB�<B'BJB�B��B��B+�WB@�B ��B?�A���A�}�B��B?MB%@1B@�B�kB@B�B� B@QB,��B��B�3@�DA��'A�g�@ÓsA��.A�D�A4�iAxUAOAʌ�A��uA�+RAy��ACbA�Z@�GgA?�A�[�A��IA�)}A���AiTPA���@�2�Ay��A��A�&�A�v�AcK�B*@��jA���A��A���AQ�A���A"[�?��?c>A��A�AA��7@�M�A:�DAD�}AT	/@�B�B�{A���C��(C��W@��yC�I�@ͱ@�A���A�y�@hA�s�A�}A7YAw5
A�A�׳A�b�A�^�Az�VAB��A�H@�A??Aŀ�AuA�z*A�.Ah0�A��)@��Ay��A�U�A��*AɁ�Ab�rB<�@��6A�|�A��SA��&AR�A���A!}|?�XV?P�hA�вA篲A�dB@��"A;H�AD�AU��@��B�A���C��[C���@���C�W�@���         �   ;   )   %   
            (            U         	      Y   +   
   +      	               2               (                     
      	               %   J            2         ;   '      #                        )   /               +   +      '                  !   =         !      $                                                                                                   %   '                  %      %                  !   3               $                                                         N5FqO8�Ov�(O��O�O���Of��N���N&�N��sO��N���O1PN]P=kO�dN�6�N/0?N���Ow��PON�)�P
�bM�1Oa{NψNC7dN-ׄO��PC��N�$:N`�OAUN*��P@�Ng#+Ohm�OI��OJ�N���O3N�TN� ;N1��N�:�O�HN�zNo�nO5�SO��N&�YN��N��O&a�  �  v  �  �  x  {  �    !  �  d  =  �  �  �  �  N  V    	�  �    �  �  �  �  "  g  �  �  �  !  5  �  `  W  �    �  S  �  �  J  b  K  �  �  D  �  �  �  �  
�  μ���D��=�`B<49X;�`B<49X;�o;��
<#�
<#�
<�/<D��<�C�<���=#�
<��
<�1<ě�<�`B=}�<�<���<�/<���<�/=o<�<�<��=#�
=o=\)=0 �=�P=�P=�w=49X=8Q�=8Q�=<j=D��=@�=]/=L��=P�`=T��=�%=�t�=��
=� �=��#=��m>   >C���������������������fcbddgt����������tgf@@BGN[gtx�����tg[NE@��������������������tpmnptu����������ttt#/;HW^_[SH;/"nmpt��������������tn����������������������������������������������������������������������������������������������������4.67BO[_daa[YOFB=644R_gt�����������rhfbR����+5CFG@5)���HSUbjn{���{undb[UIIH)5785)LOT[[hknjh[OLLLLLLLL����������������������������������������5B[dmsrhVB8'$()55BBGGFB5)''''''uwvz�������������zu����������������kjmvz�����������zmkk��������������������)..)$).6@BDB<6+)$$$$$$$$��������������������%5@FLZ^XB5)���������������:207<?HMUHE<::::::::)5BHNONLGB?5) �������������)6BOXXUOB6������#(-/1/+#"�������������������������),,*)#������������������������������������������.,-./5>HRPSRQNHC;7/.DCBHKT]abaaXTHDDDDDD����������������������������������������##$09<A<0+&# � 
!#$''&*$#
`^fhtvvtkh``````````
!#%%#
�������
�������������
	���������������������������������������������������	


��������z�����������������zz���������������������������������������������������������������������������������[�h�l�t�z�}�|�w�t�n�h�[�O�B�8�5�9�B�O�[�����0�>�@�F�D�@�4�������������zÇÓàìøù��ùïìàÓÇ�{�z�t�z�z�z�T�a�c�m�s�q�f�T�H�;�/�"����"�/�;�K�T����#�2�A�D�H�G�A�<�4�,�(�"������������Ŀѿݿ׿ѿĿ��������������������������������������������������������������ÓØàéìïòìâàÓÇÄ�~ÁÅÇËÓÓ�A�N�Z�\�g�h�h�g�^�Z�N�A�@�5�(�(�#�(�5�A�� �(�,�,�(����������������Ŀѿܿݿ���ݿѿĿ����������������Ŀľ��������������s�f�Z�M�4�+�,�A�M�f�s�����0�E�T�Y�T�I�=�0�#����������������񼤼��������¼���������������~����������Z�f�i�i�j�h�f�Z�W�N�P�S�Z�Z�Z�Z�Z�Z�Z�Z�H�K�U�a�b�l�a�U�H�E�F�F�H�H�H�H�H�H�H�H�<�<�H�M�H�H�=�<�/�$�#��"�#�/�;�<�<�<�<Óàìÿ��������ùìàÓÇ�z�m�h�n�zÇÓ�	�"�H�T�V�S�S�N�H�;�/�"��	�����������	�T�`�m�n�y�~�z�y�m�`�T�N�G�M�T�T�T�T�T�TŠŭ������������������ŹŭŐōŏœŗŕŠ�r�s�u�s�r�f�c�\�f�q�r�r�r�r�r�r�r�r�r�r�ĿѿԿݿ���ݿۿѿĿ������������¿Ŀ��������������������������������{���������a�g�k�m�a�U�I�M�U�W�a�a�a�a�a�a�a�a�a�aÇÎÓØÓÐÇÃ�z�q�z�}ÇÇÇÇÇÇÇÇ�G�`�q�r�m�c�^�T�.���	�����"�.�;�GƳ������$�0�5�$������ƧƎ�u�j�f�mƁƞƳ��������������ܻۻܻ�������������������������������������������������)�+�+�(������������������������������������������������������׾׾���������׾������������������ʾ��"�/�;�>�;�7�/�"����	��	���"�"�"�"�����������½����������������������������Y�e�r�~���������������~�r�e�[�Q�L�E�H�Y�����'�/�3�3�'� ��������������m�z���������������������z�u�m�l�m�m�m�m�����
���#�0�<�0�#��
�����������������U�b�n�y�{�|�{�w�n�b�a�U�J�P�U�U�U�U�U�U�����������������������������������������M�M�W�M�A�4�(�$�(�4�A�M�M�M�M�M�M�M�M�M�s�������������������{�~�s�f�a�f�h�s�s�׾����	��������׾ʾ����������ʾξ׼'�4�@�K�L�@�4�0�'�%�'�'�'�'�'�'�'�'�'�'ǡǭǵǹǭǫǡǔǏǎǔǝǡǡǡǡǡǡǡǡ��������
����
���������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D�D�D�E*E7ECEPEVEPEIECE7E4E*E*E*E*E*E*E*E*E*E*�x�������������û����������x�s�x�x�x�x�xE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EͼF�R�Y�\�Y�S�@�8�4�'���
����'�4�<�F  , * . B % S 8 t @ - > ! :  ; m ] 4 D 4 . 6 Z C P D 8 W Y > 6 ' X B g a I : U X 6 H U Q g L C  P E t H A  I  �  �  &  F  R  *  �  `  &  "  �  '  �  �  D  �  p  �  �  �  �  �  8  <  �  o  N  �  �  �  j  �  b  w  �  &  �  �  �  w  �  �  j  �  n  G  k  }  #  V  �  �  r  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  :  q  �  �  �  �  �  �  �  �  |  C  �  �  a  �  �  @  �  r  F  \  h  n  r  u  v  s  o  e  T  ?  '      �  �  �  I   �  �  
�    1  <    �  &  �  a  �  �  �  0  �  �    $  �  �    �  &  m  �  �  �  s  E  &    �  �  �  Y    �  2  �  L  �    +  M  e  u  t  i  W  <    �  �  y    |  �  �  �  �  �  4  U  l  x  z  q  `  F  $  �  �  �  X    �  �    (   �  �  �  �  �  �  �  �  �  �  �  �  �  }  n  ^  O  C  5           
    �  �  �  �  �  �  �  �  v  `  J  2        #  !        	     �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �    l  V  ?  )      �  �  �  P  	  �  \    �  >  �  �  �  �  	  &  @  Q  ]  c  S  1    �  �  >  �  }    �  ;  =  7  2  ,  '           �  �  �  �  �  �  u  \  B  )    �  �  �  �  �  �  �  �  �  �  �  �  f  M  <    �  �  C   �  y  �  �  �  �  u  i  ^  R  D  3    �  �  �  y  a  `  i  _  
  z  �  �  �  �  �  �  �  a  0  �  �  5  �  >  �  �    �  �  �  �  ~  y  u  �  �  �  �  z  n  b  Q  ?  *    �  �  �  N  L  K  I  E  9  -  !    �  �  �  �  �  q  V  :        �  5  8  ;  >  A  E  M  U  ^  f  o  v  |  p  d  W  J  <  +    <  �  �  �  �          �  �  �  E    �  l    �  �  n  �  �    v  �  		  	F  	{  	�  	�  	�  	Z  	2  	  �  A  c  �  �  L  "  �  �  �  �  �  �  �    ^  6    �  �  X    �  H  �  �  �  �  �          �  �  �  �  �  �  }  Z  5    �  �  q  �  �  �  �  �  �  �  �  o  T  :  !    �  �  L  �  _  �  '  �  �  �  �  �  �  s  W  >  (    �  �  �  �  �  �  o  V  =  �  �  �  �  �  �  �  t  \  D  0       �  �  �  �  �  t  T  �  �  �  �  �  �  �  �  �  �  o  `  E  $    �  �  �  �  �  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  X  I  ;  )      �  �  �  �  �  b  <    �  �  V     �  �  �  �  �  �  �  |  i  R  <  $    �  �  �  �  �  a  2    �  �  �  �  �  �  �  ^  5  &    �  �  P  �  �  0  �    [  �  �  �  �  �  o  Q  4    �  �  �  ~  Q  !  �  �    �  w             �  �  �  o  D    �  �  �  c  :    �      �  �  �    1  5  3  .  %      �  �  �  e  %  �  �  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  s  m  g  `  Z  `  R  K  @  2  *    �  �  �  �  H    �  �  F  �  K  �   �  W  L  B  8  '      �  �  �  �  �  �  �  �  �  s  Y  @  '  �  �  �  �  �  ~  p  b  R  @  0  !    �  �  �  �  i  @      
    �  �  �  �  w  R  .    �  �  �  Q  �  �  $   �   6  �  e  ?    �  �  �  m  I  "  �  �  �  s  K    �  �  P    S  M  F  B  D  E  C  ?  :  )          �  �  �  O     �  �  �  �  �    a  A    �  �  �  �  �  w  ?    �  �  	  �  �  �  �  �  �  w  c  L  3      �  �  �  �  �  a  <    �    #  5  F  I  G  E  @  8  3    �  �  t  "  �  F  �  1  	  b  R  B  ,    �  �  �  �  z  X  6    �  �  �  �  f  9    K  G  B  >  9  2  +  $            �  �  �  �  j  /   �  �  u  _  F  +  !    �  �  �  �  O    �  �  q  J  :  *    �  �  �  �  �  �  �  �  �  �  o  D    �  �  k  -  �  �  l  D  )    �  �  �  s  Q  '  �  �  �  _  -  �  �  �  [  #   �  �  �  �  �  �  �  �  z  F    �  �  2  �  o  �  y  �  o  �  �  �  �  �  f  :    �  -    �  �  -  
f  	]  +  �  �    }  �  ]    �  k    �  s    �    �  	  �    �  X    �  V  �  �  �  c  6    �  �  \  &  �  �  �  �  {  R  &  �  �  v  
�  
!  	�  	N  	  �  9  �  z    �  K  �  s  �  �  �  @  �  6  H  T  �  �  �  �  ]  �  U  
�  
  	`  �  U  �  I  �  �  t  L