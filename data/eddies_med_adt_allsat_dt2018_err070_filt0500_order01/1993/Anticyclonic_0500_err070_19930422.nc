CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�-V      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N"��   max       P|zz      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =ȴ9      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F*=p��
     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vp�����     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�e�          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >m�h      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�DH   max       B0tQ      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{�   max       B0�b      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ? ��   max       C�\�      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��7   max       C�g�      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N"��   max       P�F      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���
=p�   max       ?��/��w      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       =�      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F*=p��
     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vpQ��     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @O@           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�X@          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_ح�   max       ?���>BZ�     �  V�   
      	            8            *                                 m   A            d      2   H            "            z         	      '   
      	   	      &      
   �   9         '               %      N�M�O�fN�;;Oy`ONoN��Pc��N2��O��OZP�N��#OA� NazvNW�'ON��N��N�-oO=&�N���N#A3P5P�P{�NvU�Nŉ�N"��O���O��]O��PO��O��P�N;O�`N��PNR�)NA�^P|zzO�aOm�N��N�;zO���O��N��:N{�JN���O��O���O�
N��,O��O��OE��N�!LP�FO?.YNm^rO�%N�CcO"R�O5��N�t!��`B��j����D���49X�o���
��o��o��o��o:�o;o;�o<o<t�<t�<D��<D��<T��<T��<u<�o<�o<�C�<�C�<�t�<�t�<�t�<�t�<���<���<���<���<��
<��
<��
<�1<�j<�j<���<���<�h<�<�<��=o=+=#�
=,1=8Q�=<j=<j=L��=L��=P�`=P�`=T��=y�#=���=�-=�9X=ȴ9##).0<ISUXUQNI<0#����������������������������������������YTPT[gt|�����xtpg^[Y��������������������snpz�����������zssssXUYam����������zmjaX��������������������#)5=BINMKB5)FLN[gt{����|tge[QNFF�����
#/7AHIF</#����|x��������������������������������������������������()+)&����&)5BNQNSQNDB5) ��������������������mnz���������ztpnmmmm/(-/0<ISUbemkbUI<;0/)459<950)!��������������������3./7B[ht������tjhOB3@;:<Cgt��������ytgN@())6BO[][YOB6)((((((42/-,6<BEGLOQROJB:64NLOP[`ec[ONNNNNNNNNN����������������������������

������#/<DHRUTRJ</#3<HUit��{ngYH<(��������� �������������(360)*4/#���rpt������trrrrrrrrrr�������

��������

�������������������������������)**6>COQOHC63*))))))������6O[goh[���kmqz���������zvqqnkSTV[]bdgkotw||zutg[S\gt�������vtig\\\\\\�������������������������!+,+)%���]acefhmz������zma]]����������������������������������������
 "(./7/"������������"/;HSVYYXTMH;%#$&/<HLUUUUIH<:/-&##���������������������������
"%#
�������~�������������������  $%)+))	$)457BGHBA5)5505Bgt������tg[NB>5 )18A=6) ��������������������;:<BHRUannonnngaUH@;��������������������xnoqtxz�����������{x����)/6:6/���vppz�������zvvvvvvvv�����������������������������������y����ĳĿ����������������ĿĳİĦĞĦĥĩĳ�����������������������������������������zÇÓàæëàÓÊÇ�z�n�j�a�_�a�e�n�v�z�����	���"�'�%�"���	���������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������w�h�Z�I�V�g�������������������������������������������Ҿ׾����	���	����ʾ������������������������������������������������������ſ;�G�T�m�y���������y�`�;�"������
��;������*�*���
������������������������������������������������ƽƴƷ�����ûл׻׻лû��������������������������Z�a�f�i�f�f�Z�V�M�L�M�P�M�L�M�W�Z�Z�Z�Z�G�T�`�h�m�m�o�m�m�`�T�G�;�7�.�-�,�1�;�G�����������������������������������������
�
�����
������������������
�
�
�
���������������������������t�t�q�r�~�Ƴ������������������������ƹƳƯƳƳƳƳ�zÇÌÇÆ�{�z�y�n�j�n�s�z�z�z�z�z�z�z�z�л���'�>�S�U�Q�F�4���黡�����û˻ɻп��Ŀѿݿ���������ѿĿ��������������ݿ���������ݿۿٿڿܿݿݿݿݿݿݾ��������ʾѾʾľ�����������u����������Z�f�r�s�x�s�f�Z�O�X�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��������)�-�*�����������ùîù����������� �*�9�A�7�5�0������ܿؿܿ��EEEE*E7E<EBE>E7E2E*EEED�D�D�D�D�E�A�s���������������s�Z�A��������(�9�A�Ŀǿѿ����������ݿĿ���������������%�B�N�b�[�B��������������������������������������������������������������(�4�A�H�Z�s��������������s�f�A�"��!�(�#�#�/�9�9�;�/�#������#�#�#�#�#�#�#�����������������������������������������y�����������~�y�s�m�j�j�m�x�y�y�y�y�y�y������������ټټԼʼ������~�~������������������ĳĚčĄ�}�}āċĚĦĳĿ�������$�0�=�I�U�I�=�9�0�$�������������t�v�y�~�x�t�g�[�X�V�[�]�g�k�t�t�t�t�t�tÓàìù����������ùìàÓÏÒÓÕÓÑÓ������������������������������������ŔśŠŭŹ��������źŹŭŠŜŔŋŊňŔŔ�/�<�H�J�T�H�>�<�/�#�#�%�/�/�/�/�/�/�/�/�l�y���������������y�v�q�l�d�l�l�l�l�l�l�/�;�H�T�[�T�Q�H�;�:�/�)�$�+�/�/�/�/�/�/�C�O�\�h�l�u�~�}�v�u�h�\�R�O�F�C�;�7�C�C��#�0�?�F�H�A�0�#��
���������������
�������	�����������������������������Z�\�f�i�g�f�^�Z�M�E�F�I�M�T�Z�Z�Z�Z�Z�ZD�D�D�D�D�D�D�D�D�D�D�D�D�DzDnDiDkDoD{D���������	������ܹϹ¹��������չ��
��#�0�6�I�L�R�I�<�0�#��������������
�b�b�n�{ŁŇňŇŁ�{�n�c�b�Y�V�[�b�b�b�b�����������������������s�g�X�R�T�[�k�s���"�(��"�'� ��	�����������������	���"��!�,�,�#�!��
��������������(�-�4�:�A�B�C�B�A�4�(����� ����������(�2�2�(�$����������!�-�:�F�S�_�c�l�v�o�l�_�S�F�:�1�-�&�!�!�~�����������������������~�y�r�q�p�r�u�~�f�r�������{�r�o�f�b�`�f�f�f�f�f�f�f�f O = q @ 1 4 4 } . ? . � L 8 m & V S / M Q 6 # v { .   H  : 5 j 0 e K ^ X [ E d m 2 7 M 9 M S C * ! K  Q ] 6 $ D 2 ' V A M $    8  �  �  Z  >  	  �  �  G  :  �  �  �  p  �  �  �    �  �  F    �  �  @  5  A  <    �  /  �  W  �  �  �  z  �  "  �      �  7  �  �  �  W  _  &  �    �  �  �  �  �  �  W  �  h  �  �����49X�49X;ě�;ě�;��
=L��;D��<�9X<e`B='�<t�<T��<#�
<e`B<�9X<u<�C�<ě�<�o<���=��=��T<�t�<�9X<��
==D��=�7L=�E�=H�9=49X<�j=T��<���<�9X<�j>n�=Y�=��=C�=@�=�7L=�w=0 �=�w='�=49X=��-=�7L=aG�>m�h=��`=��-=�C�=�Q�=}�=�%=��-=��=��m=�`B=�G�B&#�B1�B"N�B	s�BYhB��B ݅BYDB��B	=B�7B4 B*�B�0BѡB+�B*�B�dB&��B��B1Bq}B	�yB.�B>�B&IBR*B��BD�B4�B�RB7uB�|B#�B�B)��B0tQB`XB �mB	=�B	�dB!�BB�~A��PB�MB,��A�DHB�A�w�B��B��B0�BY�BBIB��B��B�B�HB̻B�yB^lB�#B&<�B�B"��B	��B@�B=rBjB��B�{B	>LB��B�-B=�B�B"B -B?�B��B&��Bq�B�EB@FB	��B\&BI�B:�B�BJ�B@B�BͯB8�B�GB#��BުB*�B0�bB?�B ��B	>`B
Z'B";dB��A�|�B��B,��A�{�B?�A���B��B@�B?BãB�2B?B��B@B��B��B��B�<B��BX�@�*A���@"��A�VUA��1C�\�A�˕A�!)AS��A�U�Ag�:A�9�B�%@��[A>{%Af*CAsa�A��@�yEB<�AȨ�@�*�A{\A}�lAK�pA@�RA��A��C�o8A�UA}g�A�M�A�G9A>&A�vA�{Am,�@�FA���B	/FA��A�=�A�x�A�TvA�nA8	A��/B�)A�W\A��(A>%�C��r? ��A須A�sA�`�A���@g4A6(�A4x�@���@��@�k@���A��@�AȅkA��rC�g�A��AЀATXXA�z�AgG�A�P"B�@�N�A>�Af~Ar�A��@�BjAȅE@��A{TA~��AK��A@��Aф�A��C�u�A���A~�(A�SvA���A>��A���A��Amu@�5�Aߠ�B��A���A�z[A���A���A�A�\A�Z�B?�A�L�A�s�A=7�C��E>��7A雴A���A��|A�k�@c��A5�A4��@��@�/@�}�   
      	            8            +   	                              n   B            d      2   H            "            {         	      '   
      	   
      &         �   9         (               %                           1            '                                 -   '            %         1      /      !            7                                          !   !         %                                          !                                                                     #      !                  '                                             !         %                     NA�5O�fN�;;N�o9N�pN���O�N2��O-�UN�O15N�DxO&o.NazvNW�'O8�N��N�-oO)N���N#A3O��OW[NvU�N��PN"��OcS�Ow��O^��O�vO)�gO��DN;O$�N��PNR�)NA�^O��sODNvNӆMN��Nڟ8O�+O��NO#�NM�N���O��OP��N�ŊN��,OO�NO��O.c|N���P�FO%)INE;�O��N�CcO"R�O5��N�t!    �  �  �  �    �  j    �  �  �  )    `  y  �  6  �  �      �  v    �  �  M  �  �    6  �  �    �  E  �    �  F  �  D  J  8  �    Z  d  �  �    	�  o  �  �  	  �      �  �  W���ͼ�j����t���`B�ě�<T����o;ě�:�o<���;o;D��;�o<o<#�
<t�<D��<T��<T��<T��=0 �=<j<�o<�t�<�C�=m�h<��
<���=<j<�`B<���<���<�`B<��
<��
<��
=ix�<�/<���<���<�`B=\)<�=o=o=o=+=H�9=H�9=8Q�=�=<j=T��=P�`=P�`=T��=Y�=}�=���=�-=�9X=ȴ9*-06<BILIH<0********����������������������������������������YV[\gntx���tgb\[YY��������������������vrvz���������zvvvvvvfdelz������������ztf��������������������)5;BBGGBB:5)SPQ[gtx~tg[SSSSSSSS	
#+/26884/#
������������������������������������������������������()+)&����)5<BNMNQPNB52)"��������������������mnz���������ztpnmmmm*,/06<IQUakib_UI<10*)459<950)!��������������������:8<?EO[ht�����ythOB:RPT[^gt��������tg][R())6BO[][YOB6)((((((531/169BDFHIOOOB>655NLOP[`ec[ONNNNNNNNNN�����������������������������������#/<@LQPNHB</#(%%*/<HUdhhfa\YUH</(������������������������� 
%+1+(#
 �rpt������trrrrrrrrrr��������

��������

�������������������������������)**6>COQOHC63*))))))��������'))%����yutsz}������������|yWVX[^agttyywtrg[WWWW\gt�������vtig\\\\\\����������������������� #'))' ���]acefhmz������zma]]����������������������������������������
 "(./7/"������������""#/;HMQTTSMH;6/*",))/2<FHPNH?</,,,,,,��������������������������

�������~�������������������#$()*)'
%)5BFGB@5)5505Bgt������tg[NB>5	")067;?<6)��������������������;<EHUabnnmmfa[USHA<;��������������������xnoqtxz�����������{x����)/6:6/���vppz�������zvvvvvvvv����������������������������������������ĳĿ����������������ĿĳİĦĞĦĥĩĳ����������������������������������������ÇÓÞàæàÚÓÇÁ�z�q�n�c�i�n�zÁÇÇ�����	��� ���	����������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������q�h�j�s�����������������������������������������Ҿ��ʾ׾������������׾ʾɾ��������������������������������������������������`�m�y�z�}�~�y�s�m�`�T�G�;�4�.�8�;�G�T�`����������������������������������������������
��������������������ƿƶƹ�����ûл׻׻лû��������������������������Z�a�f�i�f�f�Z�V�M�L�M�P�M�L�M�W�Z�Z�Z�Z�G�T�`�f�l�n�m�k�`�T�N�G�;�9�.�-�.�2�;�G�����������������������������������������
�
�����
������������������
�
�
�
����������������������������w�v�t�����Ƴ������������������������ƹƳƯƳƳƳƳ�zÇÌÇÆ�{�z�y�n�j�n�s�z�z�z�z�z�z�z�z�����'�4�?�G�H�E�4�����ܻٻԻػٻ�Ŀѿݿ���������ݿѿĿ������������Ŀݿ���������ݿۿٿڿܿݿݿݿݿݿݾ��������ʾξʾ¾������������������������Z�f�r�s�x�s�f�Z�O�X�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������������(�5�7�>�5�(������޿ڿ߿��EEEE*E9E>E:E7E*EEED�D�D�D�D�D�D�E�5�A�Z�s������������s�g�W�5�(����$�5�ѿݿ������� ������ݿѿɿĿ����Ŀſ����)�5�B�N�W�\�N�B�)��������������������������������������������������������4�A�M�W�f�s�������s�f�Z�A�4�0�(�#�(�4�#�#�/�9�9�;�/�#������#�#�#�#�#�#�#�����������������������������������������y�����������~�y�s�m�j�j�m�x�y�y�y�y�y�y�������ּ������ʼ�������������������ĚĦĳĿ����������ĻĳĚčćāĀĀāčĚ������$�0�<�1�0�$������������������t�v�y�~�x�t�g�[�X�V�[�]�g�k�t�t�t�t�t�tàìïùÿ��������ùìàÓÐÓÔØÔàà��������������������������������������ŔśŠŭŹ��������źŹŭŠŜŔŋŊňŔŔ�/�<�E�H�N�H�<�<�/�&�%�*�/�/�/�/�/�/�/�/�y���������������y�x�s�m�y�y�y�y�y�y�y�y�/�;�H�T�[�T�Q�H�;�:�/�)�$�+�/�/�/�/�/�/�C�O�\�h�l�u�~�}�v�u�h�\�R�O�F�C�;�7�C�C�
��#�.�0�:�>�7�0�#��
���������������
���������
�������������������������޾Z�\�f�i�g�f�^�Z�M�E�F�I�M�T�Z�Z�Z�Z�Z�ZD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzD|D�D�D���������	������ܹϹ¹��������չ��
��#�0�3�<�H�C�<�0�#���������������
�b�n�{ŀŇ��{�n�d�b�Z�W�b�b�b�b�b�b�b�b�����������������������s�g�X�R�T�[�k�s�������	���"�&�"����	�����������������!�)�)�"�!���������������(�*�4�8�@�A�A�8�4�(����������(�������(�2�2�(�$����������!�-�:�F�S�_�c�l�v�o�l�_�S�F�:�1�-�&�!�!�~�����������������������~�y�r�q�p�r�u�~�f�r�������{�r�o�f�b�`�f�f�f�f�f�f�f�f 2 = q A . 9 # } % 3 6 ] D 8 m % V S + M Q   v { .  C  O 1 L 0 Y K ^ X 0 8 N m 0 8 M 4 O S C - ( K  Q S 5 $ 2 @ ! V A M $    c  �  �  �  �  �  *  �  m  �    �  z  p  �  �  �    m  �  F  �  �  �  �  5  �    �  �  n  �  W  �  �  �  z  R  �  �    �  !  7  c  |  �  W  �  �  �  �  �  �  �  �  d  l  :  �  h  �  �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �            �  �  �  �  �  �  �  �  �  �  �  �  t  j  _  P  >  )    �  �  �  �  �  �  v  d  R  ?  �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  H  5  "      �  �  �  �  �  �  �  �  �  �  �  �  q  ]  J  6    �  �  K  �  �  �  �  �  �  �  �  �  �  �  �  s  L    �  �  3  �  ~    �              �  �  �  �  �  �  t  O  9  ;    �  �  �    R  �  �  �  �  �  �  �  �  f  ?    �  �  '  �  �  �  j  _  T  I  @  7  3  3  2    �  �  �  �  o  J  $   �   �   �  �  �  �                �  �  �  �  e    �  e  �  �  �  �  �  �  �  �  �  �  �  �  |  h  R  ;  #    �  �  �  �  ]  i  f  V  B  9  N  |  �  �  �  z  O    �  }  %  �  =  n  K  s  �  �  �  �  �  z  f  Q  Q  d  t  o  j  d  ^  S  ?  +  #  '  '      	  �  �  �  �  �  �  �  {  c  K  +    �  �       �  �  �  �  �  �  �  �  �  �  t  O    �  |  P  $   �  `  z  �  �  �  �  �  �  �  {  j  W  C  /      �  �  �  �  ]  p  v  o  _  N  <  )    �  �  �  �  �  q  N  +  �  �  �  �  �  �  �  �  �  �  x  p  h  b  _  [  Y  W  V  U  Q  N  J  6  0  )  #          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  N  <  3    �  �  �  �  �  �  i     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  J  .     �                                 �  �  �  �  $  	\  	�  	�  
�  	      
�  
�  
h  
)  	�  	�  	x  �  O  �  �      �    k  �  �  ,  Q  o  �  �  {  b  2  �  �    u  �     �  v  p  j  e  _  Y  S  M  G  B  ;  5  .  (  !                             �  �  �  �  �  �  �  �  �      0  �  �    }  {  y  w  u  t  s  r  q  p  n  i  d  _  [  V  Q  L  $  �    b  ~  �  �  �  �  e    �  1  �  
�  	�  H  �  �  B  M  K  =  4  B  I  E  7       �  �  ~  >  �  �  6  �   �  �  �  �  �  �  �  �  �  X    �  Q  �  x    �  �    +  �    s  �    N  ~  �  �  �  �  �  b  )  �  {  �     q  �  4  �  �  �  	          �  �  �  g  '  �  �  ;  �  Y  �  �  �  �  �    5  5  +      �  �  �  �  o  >  �  �  4  �  +  �  �  �  �  �  �  �  �  �  �  �  �  �  }  s  b  M  9  %      -  Z  �  �  �  �  �  v  W  -  �  �  �  D    �  {  0          �  �  �  �  �  �  �  v  i  \  O  A  3  $      �  �  �  |  v  o  h  a  [  T  M  F  ?  7  0  )  !          E  ?  9  3  -  '  !      
     �   �   �   �   �   �   �   �   �  
�  
�  
�  
P  {  �  �  �  �  �  �  3  
�  
  	l  �  ~  Z  �  C  �  �        �  �  �  �  �  �  n  >    �  �  f  (  �  �  �  �  �  �  �  �  �  �  �  �  r  T  /    �  �  R    �  �  F  <  3  *  "    
  �  �  �  �  �  �  w  ]  ?     �  �  P  �  �  �  �  �  �  �  �  y  Z  :      �  �  �  w  ~  �  |  �    3  D  3    �  �  �  �  o  D    �  �  >    �  '  P  J  A  9  0  '      �  �  �  �  �  r  M  (    �  �  �  �  )  0  5  8  5  ,      �  �  �  �  �  l  Q  4    �  �  �  �  �  �  �  �  �  �  �  �  �  v  ^  F  +    �  m  "   �   �    �  �  �  �  �  �  x  \  =    �  �  �  �  k  ,  �  d   �  Z  U  P  H  A  ;  5  .    
  �  �  �  y  E    �  �  p  N  �  �  $  D  [  d  b  _  [  W  O  @  *  	  �  �  1  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b    �  B  �  B  �    �  �  �  �  �  �  �  o  W  =  "    �  �  �  �  �  �  m  Y  -  �  �  I  �  R  �  �    �  z  �  ;  V    �  �  C  	w    	�  	�  	�  	t  	o  	R  	"  �  �  _    �  c  �  g  �  V  �  k  K  8  n  m  ]  C  $     �  �  �  K    �    3  �  �  (  i  �  �  �  �  �  �  x  `  B     �  �  �  n  8    �  �  �  s  U  �  �  �  �  t  >  �  �  u  +  �  �  k  6  �  �  (  �    �  �  �    �  �  �  �  �  k  L  +    �  �  �  �  v  w  K    �  �  �  �  �  �  �  �  �  t  Z  A  (    �  �  �  �  '  P  �    �  �  �  �  �  �  �  �  u  ]  B  #  �  �  �  �  �  �    �  �  �  �  �  �  �  w  b  L  3    �  �  ~  C  
  �  �  �  �  �  u  S  /    �  �  w  D    �  X  �  H  �  �  t  )  �  �  {  _  H  A  J  C  0    �  �  �  |  D    �    N  y  W  .    �  �  �  �  o  Q  .    �  �  s  @  
  �  �  7  �