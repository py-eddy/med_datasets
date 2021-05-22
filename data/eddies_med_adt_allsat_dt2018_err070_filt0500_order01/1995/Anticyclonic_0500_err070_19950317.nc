CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��x���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�3�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E\(��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vo
=p��     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q`           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�ݠ          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >O�;      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�9   max       B,؆      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�=*   max       C�T�      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��/   max       C�V�      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_ح�   max       ?�,<�쿲      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       =���      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E\(��     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vo
=p��     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q`           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�           �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @   max         @      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��,<�   max       ?�+j��f�     `  U�   
      S         V      I            (   S                     $                  X   *         �                     	   j   +   L                                    (            
                  	Nu�Oe��P�3�NX�XO�;CPY��N�?8P!�N}N�~�NAw�O��P���N	XZO�OͯO�ݎOXǐOF�qP�xN��zP�NL�9OwwOgr�P]UPT%N�y$O��^P��JN��KN)j�NO�kNX$lO�ZO�N.P�P���O��IP.��O� �Oh��N*��Na3O;�N�ޔN�N3NnOx��O$�N�*O�yN])N�S-N,�uN�;N;b�M��Nu�O$3N6�KN?�Ӽ��㼓t��#�
�#�
�o�ě��D��$�  $�  :�o;D��;�o;�o;�o;��
;ě�;ě�;ě�;�`B;�`B;�`B<49X<D��<�t�<��
<�1<�1<�1<ě�<���<���<�`B<�h<��=o=o=+=+=+=C�=C�=t�=��=��=��='�=,1=,1=49X=H�9=H�9=H�9=Y�=]/=e`B=m�h=u=u=y�#=}�=�o=���fgt�����ztqmhgffffff\\__cgt����������tg\�����)Bg|�dRB)������������������������������&#
�����������).3-�������-**#/<HIKHHA</------'*0>N[gt����g[NB5*'���
����������jmuz�������zsmjjjjjj	$%�������������������):HNK6	�����������������������������������������������qopz��������������zq�������	��������������

��������
#/0675/#
��4<=OYft�������thOB94A<;;BBOS[[_\[OGBAAAAkcmnv������������xtk973<HMUUUMH<99999999��#/496/#
����������������������icbemq{�����������zi��������������������TRUamntssqnbcaWUTTTT"/;@CEKHC;/"����6K[z�zh[O)���feehtv�����xtqhhffff������������������������������������egigbgtt����tgeeeeee�������
������XV]ehtx�����~tjhfc^X__dgtxztqg__________hpnq{�����������h�������������������������+584)�����rotz}������������vr�����
#*05:80#
��;6549;<BGHIH<;;;;;;;
#%($#"
����������  ��������$)58<55+)kbnz����znkkkkkkkkkkxz�������������zxxxx����)-34."�������������������������������� ���������)5?GIJEB5)//4<HLLHH<0/////////��������������������`ajnnxz~zunkfa``````���
#!#"
������!&))5687866)):@BBCOVOOB::::::::::��������������������WY]aammz������zpmaWW
#)*#
:<<HNU`[UHC<::::::::�n�u�s�n�m�a�U�M�H�F�H�U�a�f�n�n�n�n�n�n���*�6�C�N�H�C�6�*�������������������<�b�{ŝŨŤŊ�x�n�U�
���������������/�<�H�L�H�E�<�1�/�.�&�$�/�/�/�/�/�/�/�/����������%�)�0�)������������������������
������ìàÊÀ�|ÀÐà������E7ECEPE\EaEbE]E\EPEDECE7E2E4E7E7E7E7E7E7��)�6�B�L�P�N�>�6�)�����������������ù����������ùóìæìñùùùùùùùù�a�c�m�q�x�w�m�i�a�T�T�Q�T�^�a�a�a�a�a�a�����������������������������������������Ŀѿؿܿڿ˿ſ��������m�d�d�h�y���������	�"�H�V�j�l�Z�H�/�	�������������������	�'�3�5�4�3�'�'�$�����'�'�'�'�'�'�'�'���(�4�<�B�>�4����нĽ��Ľ˽н���Z�g�s�������������������s�g�`�`�W�L�O�Z��5�N�Z�]�b�_�Z�N�A�5�(������������'�6�?�J�I�@�4�'������������`�m�y������y�m�m�`�T�G�@�;�B�G�K�T�^�`�����4�@�X�L�I�@������ݻػۻֻܻ�l�y�������������������y�r�n�l�g�l�l�l�l�(�A�N�g�����������s�Z�N�5�+��	����(�A�N�Z�d�`�Z�N�N�M�A�?�>�A�A�A�A�A�A�A�A�f�s����������̾;Ⱦ��������|�}�s�b�\�f���������������������m�h�b�`�f�m�u�y�����tčĚĦĬĳĿ������ĿĳĦčā�m�h�i�f�t���"�K�6�)��	�����������������������������$�&������������������������"�/�H�T�a�b�h�j�g�a�T�H�;�/�"�����Y������ּ��ּ������r�m�r�o�D�A�P�X�Y������������������������}��������������zÇÊÇ��z�u�n�d�k�n�s�z�z�z�z�z�z�z�z�y�����������������������y�x�y�y�y�y�y�y����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DsDpDyD{D�D��-�:�F�S�V�U�S�I�F�:�!��������!�-�m�y���������y�o�m�c�m�m�m�m�m�m�m�m�m�m���!�-�F�_�f�n�{�}�u�_�F�!���ֺѺӺٺ�ù����������������ùàÓÇÀÂÇÓàìù�������������������������x�u�v�{���������N�g���������������z�W�N�A�;�8�1�0�5�A�N�s�������������������������s�f�^�\�f�s����(�5�7�5�(��������������������������������~����������������������������������������������y�w�u�v�y���N�[�g�p�t�u�t�t�g�[�[�N�L�H�N�N�N�N�N�N�/�<�G�F�A�<�/�#�$�+�/�/�/�/�/�/�/�/�/�/�
����������
�� � ���
�
�
�
�������׾�����׾ʾ������������������6�C�O�\�d�r�u�x�y�u�h�\�O�N�C�;�6�1�3�6���������������������������������������Ǝ����������������ƳƧƎƁ�}�u�s�u�zƁƎ���������������������������������������!�(�-�:�F�N�S�_�l�l�_�S�F�:�.�-�!���!�zÀÇÇÇ�z�w�n�e�a�`�a�n�y�z�z�z�z�z�z����!�!�#�!�����������������e�k�r�~�������������~�r�q�e�e�b�e�e�e�e����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ŔŠŭŸŹ����������ŹŭŠŞŖŔœŏŔŔǡǬǭǱǭǩǡǔǑǔǗǛǡǡǡǡǡǡǡǡ���������������������������������������� ? H * = Z O 1  K H ( <   S G M & @ K M - 4 9 _  + R a . D < } q M $ $ = 5 J  @  s H  G P � T G j * ; � = $ z W c B [ P  �  �  �  |  �  �  �  �  �  �  Z     W  A  �    @  �  �  r  �  f  c  *  �  s  �  �  +  d  �  j  �  d  d  d  J  �  1  �  �  �  �    �  �  �  �  '  p  �  �  ]     J  �  �    �  H  r  d�#�
�o=�t���o<D��=���<#�
=�t�<49X<o;��
=0 �=�-<o<�h<ě�<�`B<���<�`B=0 �<�9X=t�<e`B=��=�w=�;d=}�<�`B='�>O�;=o=�w<��=+>"��=L��='�>O�=���=�G�=aG�=P�`=0 �='�=Y�=H�9=Y�=<j=��=�O�=aG�=�9X=�O�=y�#=y�#=�C�=�o=}�=�t�=��P=�7L=�/B	�B
!:BWyB�OB�/B��B�'B��B�B NB�tB�BB!��B ��B�B�B#_�BTgB�yB�BJDB�B�*B�zBG�B��BfZA�9B�rBDMBB+�B	�B�`B�B	e�B��B!��B�!B hB$�~A�^�B$�=B,؆B��BF�B<uB)�B �BGxB�0B�B*�nBÄB$�B*%BbB?�A���B6�B\7B	�0B	��B�B��B6_B�hB��BK�B��B RBB��B�OB!IjB �rB ?�B~�B#T�BA�B@B��B��B�%BCDB��B?�B�GB?�A���B�*B?�B�~B5�B	��B��B�B	U�B�TB!��B��B
��B$��A�PyB%,�B,��B��BAjB��B{%B�)B��BB��B*�B��B$�,BBB��B<|A���B=B@A��A�p�A�1JA���A��]A�ƹC���AӚoA��A��>A�̥Ar�A��?�=*A1�`A�-A���@�Ahܷ@�0IA�4A��A��ZAJ��Ao��AߛA�"�A���A�@@�P�AIbA�9 Aq2�AJ�	C��1@x0Al�n@t)A�u�A� �A�{�AFkpA�hc@�8�A �7A��A�mA�_+AO3B��A�c�B�8A�{K@��A���A��@-�@!�C�T�A�P�B��A�vOA�qkA���A�0A°�A���AΕC��AӋ�Aͱ�A��
A�ޛAqEA��D?��/A0�A��KA��;@ÄcAiT@��wA�	A�|�A��0AK�Aq�A߁A�mXA��A��@�/AH�	Aȁ�Asb�AI�wC���@{y�Al�@c�A�r�A���A��AD��A�Bh@�7A�A��#A�A�d�AP�BC�A�T�B�KAч�@{��AǊIA		�?�7@#�C�V�A���B{�A��   
      S         V      I            (   T                     %                  Y   *         �               �      	   j   +   M                  	                  (                              
         ?      !   1      '            !   =      #   !            )      '            %   +         ;                        9      ,                                                                           5                              5                     )      %               !         #                        5                                                                        NGOe��P��NX�XO=�5O�H�N�?8Ohe�N}NMȳNAw�O�޽P�RN	XZO��O�QNO]H[O<�9N�3�P�xN�'O���NL�9O+�RO!b�OqTdO��N�y$O��^O��N��KN)j�NO�kNX$lO&<N�8N.P�P�^O5�,O���O� �Oh��N*��Na3O;�N�ޔN�N3NnOx��O$�N�*O��N])N�S-N,�uN�;N;b�M��Nu�O$3N6�KN?��  v  Z  �  �  �  	<  ;  �  5  P  \  �  �  �  �  f  �    �  �  E  R  �  �    
�  �  x  l  M  W  �  (  L  �  �  �  	�  �  Q  �  �  �  3  �  s  �  �  �  �  �  �      G  �  �  �  �  �    ���t���t�<#�
�#�
�o<�`B�D��=C�$�  ;o;D��<�o<ě�;�o<t�<t�<#�
<o<T��;�`B<o<T��<D��<�1<ě�=e`B<�<�1<ě�=�{<���<�`B<�h<��=��T=t�=+=49X='�=�C�=C�=t�=��=��=��='�=,1=,1=49X=H�9=H�9=P�`=Y�=]/=e`B=m�h=u=u=y�#=}�=�o=���sniht|����wtssssssss\\__cgt����������tg\����)5de[WPB5����������������������������������
	�����������������-**#/<HIKHHA</------<878@BN[grtvwrg[NMB<���
����������unxz�������zuuuuuuuu	$%���������������������)6?D>6���������������������������������������������tqsz��������������zt��������	�����������

�������
#(/130/#
��4<=OYft�������thOB94B=<<BEOQZ[^[[ODBBBBBmmqry������������zom973<HMUUUMH<99999999	
#/13463/#
���������������������}~������������������������������������TRUamntssqnbcaWUTTTT"/;@CEKHC;/"�����)6<LNLE6)�feehtv�����xtqhhffff������������������������������������egigbgtt����tgeeeeee�������

������[Y[`ht�����thb[[[[[__dgtxztqg__________uwz������������{u�������������������������%)+*)!���rotz}������������vr�����
#*05:80#
��;6549;<BGHIH<;;;;;;;
#%($#"
����������  ��������$)58<55+)kbnz����znkkkkkkkkkkxz�������������zxxxx����)-34."�������������������������������� ���������	)5>FHIIB5(//4<HLLHH<0/////////��������������������`ajnnxz~zunkfa``````���
#!#"
������!&))5687866)):@BBCOVOOB::::::::::��������������������WY]aammz������zpmaWW
#)*#
:<<HNU`[UHC<::::::::�H�U�a�n�r�n�k�a�U�O�H�G�H�H�H�H�H�H�H�H���*�6�C�N�H�C�6�*����������������
�0�I�b�{ŏŒŐň�b�U�4��������������
�/�<�H�L�H�E�<�1�/�.�&�$�/�/�/�/�/�/�/�/�������������������������������������������������ùìàÛÕÔÚàù����E7ECEPE\EaEbE]E\EPEDECE7E2E4E7E7E7E7E7E7������)�+�.�'��������������������ù����������ùóìæìñùùùùùùùù�T�a�m�o�u�q�m�l�a�V�T�R�T�T�T�T�T�T�T�T�����������������������������������������������ĿſĿ����������y�q�m�l�q�y�|�����	�"�/�H�R�]�_�X�@�5�"�	���������������	�'�3�5�4�3�'�'�$�����'�'�'�'�'�'�'�'���(�1�4�:�4�(�����ݽ˽нӽݽ�����Z�g�s�������������������s�m�g�Z�W�O�S�Z��(�5�A�N�U�Z�X�T�N�D�A�5�.�(��������'�2�;�@�H�C�@�4�������������p�y�|�z�y�r�m�d�`�T�G�F�@�G�J�Q�T�`�l�p�����4�@�X�L�I�@������ݻػۻֻܻ�l�y�������������������y�y�o�l�j�l�l�l�l�5�A�N�g�����������s�Z�N�5�.�����(�5�A�N�Z�d�`�Z�N�N�M�A�?�>�A�A�A�A�A�A�A�A�������������Ⱦʾ����������������������������������������������y�m�l�h�k�m�y����āčĚĦĳĻ��ĽĶĳĦĚčāĀ�y�y�|Āā�����"�/�8�7�-�"��	�����������������������$�&������������������������"�/�H�T�a�b�h�j�g�a�T�H�;�/�"�����f�r������ǼҼռѼʼ�������k�^�W�X�]�f������������������������}��������������zÇÊÇ��z�u�n�d�k�n�s�z�z�z�z�z�z�z�z�y�����������������������y�x�y�y�y�y�y�y����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D�D�D�D��-�:�>�F�Q�R�F�@�:�-�!�!���!�-�-�-�-�-�m�y���������y�o�m�c�m�m�m�m�m�m�m�m�m�m��:�F�Z�w�y�q�n�_�F�!���׺Ժ׺����àìù����������ùìàÓÐÇÄÇÒÓÜà�����������������������������������������N�g���������������z�W�N�A�;�8�1�0�5�A�N�s�������������������������s�f�^�\�f�s����(�5�7�5�(��������������������������������~����������������������������������������������y�w�u�v�y���N�[�g�p�t�u�t�t�g�[�[�N�L�H�N�N�N�N�N�N�/�<�G�F�A�<�/�#�$�+�/�/�/�/�/�/�/�/�/�/�
����������
�� � ���
�
�
�
�������׾�����׾ʾ������������������6�C�O�\�d�r�u�x�y�u�h�\�O�N�C�;�6�1�3�6���������������������������������������ƁƎƧ��������������ƳƧƚƎ��w�u�w�~Ɓ���������������������������������������!�(�-�:�F�N�S�_�l�l�_�S�F�:�.�-�!���!�zÀÇÇÇ�z�w�n�e�a�`�a�n�y�z�z�z�z�z�z����!�!�#�!�����������������e�k�r�~�������������~�r�q�e�e�b�e�e�e�e����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ŔŠŭŸŹ����������ŹŭŠŞŖŔœŏŔŔǡǬǭǱǭǩǡǔǑǔǗǛǡǡǡǡǡǡǡǡ���������������������������������������� < H 3 = H = 1 ' K N ( & * S : Q  B J M - 7 9 (   N a . + < } q M   % = 0 6  @  s H  G P � T G j , ; � = $ z W c B [ P  i  �  �  |  �  �  �  �  �  x  Z  
  ,  A  %  V  �  �  (  r  �    c  k  V  �  �  �  +  0  �  j  �  d  S  �  J  4  �  �  �  �  �    �  �  �  �  '  p  �  �  ]     J  �  �    �  H  r  d  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  p  r  u  r  m  e  \  O  @  1  !    �  �  �  �  �  l  L  -  Z  L  J  K  F  8  *          �  �  �  �  X  "  �  �  �  �  ?  ~  �  �  �  �  �  R  �  �  7  �  �  j  S  
  �  �  �  �  �  �  �  �  �  �  �  �  z  m  `  R  E  7  *      �  �    &  G  f  x  �  �  v  f  V  D  ,    �  �  �  [    �  �  ^  �  �     j  �  		  	4  	:  	,  	  �  �  )  �  #  |  �  �  �  ;  .      �  �  �  �  �  n  F     �  �  |  ;  �  �  x  K  p  �  �    2  B  X  {  �  �  �  �  n  C  �  �  �    �     5  2  .  &        �  �  �  �  �  �  �  {  i  X  G  �  �  J  L  N  M  E  =  /      �  �  �  �  �  �  u  _  H  1    \  \  \  [  [  [  [  [  Z  Z  Z  Z  Z  Z  Z  Z  Z  Z  Z  Z    A  d  �  �  �  �  �  �  �  �  [    �  q    �  8  �  �  �    ]  �  �  �  �  �  �  T    �  �  b  �  z  �    8   [  �  �  �  �  �  �  �  �  �  �  �  �  y  e  P  9  !  	  �  �  v  �  �  �  �  �  �  �    o  X  <  '    �  �  �  I  �  j     (  O  c  `  X  K  9  "  
  �  �  �  �  �  �  ~  H    �  �  �  �  �  �  �  �  �  �  �  �  �  }  Z  ,  �  �  k    �               �  �  �  �  �  p  [  O  -  �  �  ^    �  �  �  �  �  �  �  �  �  �  �  r  F    �  �  C  �  �  �   �  �  �  �  �  ~  ^  >    �  �  �  �  O  Y  3    �  �  �  �  :  B  7  (      
  �  �  �  �  �  �  �  �  l  S  /  �  �  G  N  P  J  >  -    �  �  �  s  ^  W  V  R  >  
  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  m  d  \  �  �  �  �  �  �  �  �  �  �  �  |  \  2  �  �  �  -  �  -  �               �  �  �  �  l  C    �  �  D  �  0   k  �  	t  	�  
)  
v  
�  
�  
�  
�  
�  
�  
�  
b  	�  	�  �       '    =  �  �  �  �  �  �  �  �  �  �  z  j  T  0  �  �    V  �  x  g  V  D  0          �  �  �  �  �  �  ]  6   �   �   w  l  l  g  Z  G  *    �  �  x  8  �  �  �  �  �  S  �     I  
}  [      �    I  I  2      �  �  X  �  �  
�  	8  �  �  W  M  C  :  4  0  +      �  �  �  �  �  c  ?     �   �   �  �  �  �  �  �  �  �  �  �  �    �  0    �  �  �  }  [  9  (               �   �   �   �   �   �   �   �   �   �   �   �   �   �  L  I  F  C  @  <  9  6  3  0  )      
   �   �   �   �   �   �    n  �  �  ,  u  �  �  �  �  Y  �  {  �  n  �  �  	�  &  �  p  p  v  �  �  �  }  r  c  S  B  .    �  �  �  �  L    M  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  V  4    �  �  	a  	�  	�  	�  	k  	  �  x  �  �  {  ?  �  t  �  �      z  =  9  7  8  >  �  z  Z  E  6    �  �  �  P     �    �  5  [  0  {  �  �  �    !  >  P  N  6    �  g  �  p  �  �    �  �  �  �  �  �  �  �  �  �  �  �  q  V  7    �  �  q  ;    �  �  �  �  �  �  j  N  0    �  �  �  v  I    �  �  V  $  �  �  �  �  �  p  Z  D  .    �  �  �  �  �  |  \  ;     �  3  )           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  d  O  9  #    �  �  �  �  c  <    �  �  �  v     �  s  m  f  \  P  D  6  (  !            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �    o  ^  N  =  /  $        �  �  �  �  �  �  �  �  v  U  2    &  (       �  �  �  z  b  #  �  �  7  �  �  �  �  �  �  �  n  I    �  �  w  5  �  �  >  �  �  d  �  �  �  �  �  �  �  �  �  s  g  ^  T  K  B  9  2  8  >  C  �  �  �  �  �  �  |  c  D    �  �  \    �  ;  �  #  �  �    
    �  �  �  �  �  �  y  R  #  �  �  �  0  �  u    �      �  �  �  �  �  �  �  �  u  d  N  8  &      
  	  	  G  >  6  .  %      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  r  g  Y  J  6        �  	  M  _  l  �  �  �       	  �  �  �  �  �  r  I    �  �  �  �  s  S  �  �  �  �  �  �  �  �  �  {  q  g  \  R  H  >  4  )      �  m  K  &  �  �  �  �  Q    �  �  �  _  -  �  �  �  Z  /  �  �  �  �  c  @    �  �  �  �  b  B  "    �  �  T  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  z  m  a  T  �  �  �  y  d  F  '    �  �  �  N    �  �  U    �  �  P