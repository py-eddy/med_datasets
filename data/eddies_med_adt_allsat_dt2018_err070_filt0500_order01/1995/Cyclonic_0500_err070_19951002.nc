CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��Q��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�t   max       PQ��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��l�   max       <���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @F*=p��
     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @v
=p��     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P@           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�<�           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       <o       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B/��       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�i�   max       B0;       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?ĵL   max       C��       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C��6       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          <       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          /       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�t   max       PQ��       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?ΐ��$tT       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��x�   max       <e`B       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F(�\)     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @v~�Q�     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @M�           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�<�           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         AF   max         AF       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?ΐ��$tT     P  X�                                    +   &                                    !   !                  	   %      .                  :      
                     <                        
   -                  N�diN��$O �N�g�N#!NЩO.}�P,�tO�Z�O�� Ot�O��O���P��NP�.OY�O�E)O{��N�%�O9�O �SO�gNe*QNJ�ZPQ��O���N�	�N��O���N�-�O��N�j
OAg�O]O�}N]�LNr%N�I�OD<NpMO���N���O ��NKo?O	?OZ�O���N��cO:�}O�e�N`�O�	O�N6�+Pb�N��?O��8N��&O�ZO�dN�tN1IO��~NMz�N�ސ<���<e`B<e`B;��
;��
;�o�D�����
�ě��ě��49X�D���T���T���e`B�u��o��C���C����㼬1��j�ě��ě��ě�������h���o�+��P��P��P����w�#�
�#�
�',1�0 Ž0 Ž0 Ž0 Ž8Q�@��@��H�9�P�`�P�`�Y��Y��aG��e`B�ixսixսu�u��C���C���\)��hs���w���T��l���l�������������������������������������������������������������������������������

�������������������������������#/<@LOHA</,%#LNg���������ti_Z\ZRLNXdm�������~zma\TKJN*6CHNQQOIC6����������������������
$/<HU`UGH<;/# ���������������������������������yz��������zyyyyyyyy
.6BFORUSOB6)
	
krz������������zmkjk"/;=>B=??B;/"cnqwvz����������zync�������������������adgkmz~�����zmia^^a��������������������	

						������

���������#0Ib{�����{b<0:BO[t������tgVNB5*-:@BN[]be[WNB956@@@@@@�������������������������	�������������������������mz���������������}zmJN[gpt����tlgf[NECJJ��#(&#
���������������������������������������������������������������������������������������������������������������������;<AHJUafdaZUHH?<;;;;nz�������������nhdgn@BEO[`hqsh`[POFB@A@@������������������������������������������������������������������
���������Sbn{����������{n]YRS���

��������������������������������#02<IU]ege^UI<5-##/018<INUVUQI<0//////������������������������������������������������������������(2B[gqy{��w{�t[B/(&().-+*.)269B[iuxxztqh[TOI?52ntx��������trlnnnnnn�������������������������� ���������FHRUYYUHGAFFFFFFFFFF��������������������GIU\z�������voaUNJKG#/441/# ./3<EHLNOIHH<95/.-..�|�|¦°²¶·²¦�#�����#�/�<�H�L�U�V�U�H�<�/�#�#�#�#����������������������������������������������������������������������������������������������������������������������������
����$�������������)�&�$�&�'�)�6�B�O�[�h�i�h�[�Y�[�O�B�6�)���������������Ŀ����$�$�����ݿѿ����6�*�����*�6�C�O�\�u�~��}�u�l�O�C�6�,��	��������	�"�.�;�C�G�I�K�K�G�;�,�_�S�I�F�E�F�P�S�_�l�x����������x�l�`�_�H�D�/�#�
����4�U�^�n�}ËÇÂÁ�z�U�H�;�/�#�����"�;�H�T�a�k�q�t�u�n�a�H�;���~�{�{����������������������������������������������������������������������ҿ������������������Ŀѿ׿����ܿѿĿ������ݿϿĿ������ѿ����
��&�.�(���	���������	��"�/�;�H�P�V�T�K�H�;�/��	�����ܻлɻ��������ûлڻܻ���������ìàØÔÐËÇÓàçìù����������ýùì�)�����������)�6�9�B�E�O�O�M�B�6�)�ʾǾ����������¾ʾ׾������ ����ݾ׾������������������������������������������������������������������������������������s�\�P�F�A�B�H�Z�s������������������������m�b�^�f�z�������������������������������������������������������������������ʼɼ������������ȼʼּ�����ּʼʼʿ������Ѿ׾�	���"�.�;�A�>�5�.�"�������������������������������������������������������$�0�=�@�B�H�I�A�0������������������	�������	�����D�D�D�D�D�D�D�D�D�EEEE*E2E.E*E EED��ѿɿĿ¿ȿ̿ѿݿݿ��������������ݿѼ0�(� ��&�@�Y�f�r�������������r�Y�M�@�0�#�"��
��
��#�0�<�@�<�0�/�#�#�#�#�#�#ĿľĳĿ��������������������ĿĿĿĿĿĿ�ɺú������������ɺκֺ������ֺɺɺɺɿy�p�m�`�^�X�\�`�m�y�������������������y�����������������������������������������H�>�:�<�A�I�T�a�m�z���������������m�T�H���|�x�r�u�x�~���������������������������3�,�2�3�;�@�B�F�L�Y�_�e�i�l�e�Y�U�L�@�3�b�W�Z�b�i�o�r�{ǃ�|�{�o�b�b�b�b�b�b�b�b��������������������)�,�,�)����������������������������������������������-�'�%�)�-�6�F�S�l�x�����������x�S�F�:�-�����������������������������������������x�q�k�`�\�\�_�l�x���������������������x�ܽҽݽ�޽�����4�A�H�M�J�A�4����ܽ����������������������������������������#���
����
��#�<�I�P�^�_�]�U�I�<�0�#ŭūŠŔőœŔūŭŮŹ��������������Źŭ�O�H�O�W�[�h�p�k�h�[�O�O�O�O�O�O�O�O�O�O������������������*�6�C�Z�G�D�6�*���A�=�5�4�5�:�A�N�Z�g�s�y�v�s�g�\�Z�N�A�A���}����������������������������������������������������������������������������ā�m�g�k�xāĚĦĳĿ������������ĿĳĚā�������ʼͼּ�����
��������ּʼ�����������������������������������������������������������������������������׾ʾ������������ʾ׾��	���������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFFFFF$F1F=FJFJFJFDF=F1F'F$FFF ! r I 2 8 V / 7 D P 0 /  & 3 + h J | U P 1 \ v 2 b + G H < � X P 6 3 d r Z % V % 7 Y 5 O S g H ; 7 @ 0 I T 0 R Z 1 ( c < + 4 E c      �  M  �  D  T  u  �  �  �  6  �  �  r  m     x      �  ~    �  �  �  T  �  �  (  �  b  %  �  @  *  �  �  �  ;  �  )    8  R  _  �  �  �  �  �  �  5  5  ^  J  �  �  �  �  W  ,  J  =  \  ���o;ě�<o�o��o��o��9X�\)�������ͼ�C��]/�L�ͽ\)���
�C����+��j���t��49X����h�ixսq���\)�'T���49X�u�8Q콕���+��{�8Q�<j�ixսY��D������Y��Y��P�`�u�}󶽍O߽m�h�����`B�e`B���
��%�y�#���T������-���-��l�����w��9X��G��J�oBK�B��B�B|�B#�qBKKBOsB
M�A�7NB/��B��B3+B��BٮB�B{B�A��B��B�A��B�|B�B#�LB'�B		�B+%B!0BܨBg`Bz B	�B�TB1%B�$B�	BtB#;nB*�sB��BTkB~B ��B*�B_�B�IB(��BDB T�B&�1B&rLB�B@�B� BL�B�DB+B
G<BB-dlB{B{�B�CB��B��BBB�*BhAB�B$>4BE�BA�B
�!A���B0;BD]B�+B>B�DB�B��B ��A�i�B��B��A��kB,B�B#��B'= B	@B�TB!C'B��B��B{�B	)>B�=BAB��B�BE�B#>OB*�4B��BD�B�xB +�B=�B~�B�B(�B~�B ��B&�LB&@�B6UBY-B��B�B��B>�B
@�BAB-BTB��B�)BSPB��B��A��A��.A��A��8@��AԹ�A�gA{�B �1A^��@���A�M�A��qA�d9A���Ax�[A��A��%@�y�A� �A�EoAS�WA�O�A��&A��=A�yAJM�@�}pA]�A���B	��AY`~C�U�A};@��A�mA�: @7�HAm�YA��A��@�E?ĵLB�WA��AA�� @�TLA�"�@�f�A5&�A#��A��A�*aAڑlA�m�A�,A���A��CA�Q�A�AsKaBS�AS��C�w+C��A��SA�)A�2A�q<@�A�k�A�z�AzArB �#A\��@�AĈ�A�{�A�p�Aφ3Ax�A�~�A��0@�Y(A̔�A�rAT]�Aп4A�~�A���A��`AJ� A
A]%A���B
E�AY�)C�S�A~��@��,AꀞA�i@4hAlm�A���A���@�$�?��B�zA��GA��U@���A�e�@�bA6��A!�A��A���A��A���A���A�GA�ZA�G�A��As	�BE	AT��C�vOC��6      	                              +   &                                    "   "                  	   %      /                  ;                           <                        
   .                                          -            '      %         %                        /   '               %            #                  #                  !                        '      !      #                                          )                  %                                 /   '                                                                                                   #                  N���N��$N�[gN4��N#!NЩO��P	�:ODjO���Ot�O�TO1�P��NP�.OY�O��OR�EN���Ov�O��OK�tNe*QNJ�ZPQ��O��UN�	�N��O��N�-�O1��N�j
OAg�N���O�5N]�LNr%N_�O �NpMOe�N���O ��NKo?O	?O �`Ot�>N��cO:�}O5�?N`�OCd*O�N6�+O���N��?O�H6N��&O�O�dN�tN1IO��~NMz�N�*T    ,  y  =  D  L  J  +  w  S  �  �  �  �  m  �  �  o  �  Y  0  1  `    �  ]  0    �  �  1  F  	T  g  n  �    P  �  �  �  �  �  �  �    �  f  �  �  �  �  �  �  }  6  8  �  �  ~      �  P  �<T��<e`B<T��;o;��
;�o�ě��#�
�e`B��`B�49X�������e`B�e`B�u��1���㼓t���1��9X�����ě��ě��ě���`B��h�����+�,1��P��P�#�
�]/�#�
�#�
�49X�0 Ž0 Ž�+�0 Ž0 Ž8Q�@��L�ͽP�`�P�`�P�`��hs�Y��y�#�e`B�ixսy�#�u�y�#��C���O߽�\)��hs���w���T��l���x����������������������������������������������������������������������������������

�������������������������������#/<<HILHE?</+#U[gt��������|uga`_WUPTW`amz}��|zvnmaWTQP*6CGMPPPMFC6����������������������"#./3<@CDH<9/# ����������������������������������yz��������zyyyyyyyy
.6BFORUSOB6)
	
uz}�������������{wou!"/4;?;==?;/"xz���������zuxxxxxxx��������������������_abehlmz~�����}zma__��������������������	

						������

���������#0Ib{�����{b<05;BQ[t�����tgTNB5++5@BN[]be[WNB956@@@@@@������������������������
�������������������������������������������������JN[gpt����tlgf[NECJJ��#(&#
��������������������������������
���������������������������������������������������������������������������������������;<AHJUafdaZUHH?<;;;;uz�������������zvrru@BEO[`hqsh`[POFB@A@@��������������������������������������������������������������������
�������UVbn{���������{vl`\U���

��������������������������������)0<IU\``]USI<;30.*()/018<INUVUQI<0//////������������������������������������������������������������/6BN[hsomklg[NB3+),/).-+*.)6B[htxxyvsph[UOMJE@6ntx��������trlnnnnnn�������������������������� ���������FHRUYYUHGAFFFFFFFFFF��������������������GIU\z�������voaUNJKG#/441/# //1<DHKMNH<95//-////¦©°±¦�#�����#�/�<�H�L�U�V�U�H�<�/�#�#�#�#����������������������������������������������������������������������������������������������������������������������������
����$�������������)�&�(�)�)�5�6�B�I�O�[�d�b�[�S�O�B�6�)�)���������������Ŀѿ���������ݿѿ����C�:�6�*�)��&�*�6�C�O�U�\�h�i�g�\�Y�O�C��	��������	��"�.�;�?�E�G�I�J�G�;��_�S�I�F�E�F�P�S�_�l�x����������x�l�`�_�<�<�/�/�#� �#�)�/�<�H�U�_�a�i�e�a�U�H�<�;�7�/�.�/�2�;�C�H�T�V�a�g�l�l�c�a�T�H�;������|�|���������������������������������������������������������������������ҿ������������������Ŀѿ׿����ܿѿĿ�������ԿĿѿԿ����������������	���������	��"�/�;�H�M�T�P�G�;�/��л˻û����ûлػܻ����ܻллллл�ìàÚÖÓÍËÓàäìõ����������ûùì�6�,�)���� ������)�6�8�B�D�N�J�B�6�ʾƾ����������žʾ׾�����������׾������������������������������������������������������������������������������������s�\�P�F�A�B�H�Z�s���������������������������m�d�`�h�z�����������������������������������������������������������������ʼɼ������������ȼʼּ�����ּʼʼʿ	���������	��"�.�3�6�.�.�"��	�	�	�	������������������������������������������
�����$�(�0�=�=�A�E�E�>�0�*�$���������������	�������	�����D�D�D�D�D�D�D�D�D�EEEE*E2E.E*E EED��ѿοĿÿɿͿѿݿ��������������ݿѼ@�;�4�1�-�,�4�@�M�f�r��������v�f�Y�M�@�#�"��
��
��#�0�<�@�<�0�/�#�#�#�#�#�#ĿľĳĿ��������������������ĿĿĿĿĿĿ�ɺź��������ɺʺֺ����ֺɺɺɺɺɺɿy�s�m�`�_�Y�^�`�m�y�������������������y�����������������������������������������T�M�H�F�F�K�T�a�m�z�������������z�m�a�T���|�x�r�u�x�~���������������������������3�,�2�3�;�@�B�F�L�Y�_�e�i�l�e�Y�U�L�@�3�b�W�Z�b�i�o�r�{ǃ�|�{�o�b�b�b�b�b�b�b�b��������������������)�,�,�)�����������������������������������������������:�-�(�'�*�-�9�F�S�_�l�x���������l�S�F�:�����������������������������������������x�q�k�`�\�\�_�l�x���������������������x�����������(�4�?�A�F�B�A�4�-�(��������������������������������������������#����
�	�
��#�<�G�I�U�W�U�R�I�<�0�#ŭūŠŔőœŔūŭŮŹ��������������Źŭ�O�H�O�W�[�h�p�k�h�[�O�O�O�O�O�O�O�O�O�O�����������������*�6�B�@�C�@�6�*�����A�=�5�4�5�:�A�N�Z�g�s�y�v�s�g�\�Z�N�A�A��������������������������������������������������������������������������������ā�m�h�k�yāĚĦĳĿ������������ĿĳĚā�������ʼͼּ�����
��������ּʼ�����������������������������������������������������������������������������׾ʾ������������ʾ׾��	���������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFFFFF$F1F=FHFCF=F1F&F$FFFFF 2 r F 6 8 V - 6 5 S 0 #  & 3 + d F c X P % \ v 2 d + G 8 < ` X P 4 2 d r Q   V / 7 Y 5 O O ^ H ; & @ 1 I T # R U 1 ( c < + 4 E h    �  �  �  A  D  T  )  �  P  h  6  @  s  L  m     [  �  �  �  U  �  �  �  �  9  �  �    �  �  %  �  !  #  �  �  o    �  �    8  R  _  y  -  �  �  |  �  �  5  ^  �  �    �  �  W  ,  J  =  \  �  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  �  �                  �  �  �  �  B  �  T  �  b  �  ,  +  *  '  %  !      �  �  �  �  �  Y  0  �  �  O    �  m  q  u  y  u  p  k  e  `  [  V  Q  L  G  A  <  6  .  '    #  *  1  6  8  :  ;  <  <  >  ?  @  ?  >  -    �  �  y  A  D  ?  :  6  /  '         �  �  �  �  �  U  )   �   �   �   o  L  I  F  C  A  E  I  M  R  X  ^  d  _  R  D  7  #    �  �    G  I  J  B  2    �  �  �  r  @    �  �  �  T  	  �  ^  �    "  *  #    �  �  �  �  �  }  R    �  �  <  �  �  P  *  @  T  d  s  u  v  v  w  t  n  b  N  '  �  �  t  $  �  �  O  S  P  D  6  '      �  �  �  �  �  o  A    �  �  h    �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  g  Y  m  �  �  J  �  �    *  D  Z  s  �  �  �  k  n  h  M    �  |  �  \  ;  k  �  �  �  �  �  �  �  �  �  �  v  ;  �  �  X  �  x  �  �  �  �  �  �  y  [  :    �  �  �  �  j  F  "    �  �  t  m  j  f  b  [  R  J  A  7  -  !      �  �  �  s  P  ,  	  �  �  �  �  �  �  �  �  x  m  a  O  7    �  �  �  g  8    �  �  �  �  �  �  w  [  <    �  �  �  o  7  �  �  �  9  �  O  e  m  l  a  M  5    �  �  �  �  a  .  �  �  �  3  �  _  �  �  �  �  �  �  �  �  �  �  �  �  �  t  g  [  J  #  �  �  K  T  Y  X  P  ?  (          �  �  �  a  !  �  �  Z    /  /  )    �  �  �  �  X  *  �  �  �  r  M  "  �  �  /  �    $  0  /  '      �  �  �  �  �  p  T  /    �  �  W  �  `  R  C  4  #       �  �  �  �  �  �  q  M  *    �  �  �              �  �  �  �  �  �  �  {  ]  ?    �  �  {  �  r  R  /    �  �  �  k  B    �  �  �  N    �  z  .   �  X  \  Y  O  <  #    �  �  e    �  �  e  ;    �  f  �    0  *  $          �  �  �  �  �  �  �  �  �  �  e  C  !    �  �  �  �  �  �  �  �  �  �  �  j  I  +        �  �  �  �  �  �  w    �  �  �  �  �    h  A    �  Q  �  i   �  �  �  �  �  �  �  �  �  s  U  6      �  �  �  �  �  T    s  �    !  .  )      �  �  �  �  �  d     �  g  �  ;  �  F  ;  1  "    �  �  �  �  �  �  y  a  G  -    �  �  �  �  	T  	&  	  �  �  A  �  �  �  �  s  [  :    �  x    �  6     a  f  e  [  J  1    �  �  n  +  �  �  �  W     �  �  �  T  =  A  I  J  L  b  m  g  V  <    �  �  V  �  o    �  �  �  �  �  �  �  �  �  w  c  O  :  $    �  �  �  �  �  �  �  �       �  �  �  �  �  �  �  �  u  b  O  R  �  �  �  �  �  �  N  O  O  P  O  K  D  4      �  �  =  �  �  E  �  �  n  $  �  �  �  �  �  �  �  �  �  �  �  �  u  Y  =    �  �  �  �  �  y  m  b  U  B  0    	  �  �  �  �  �  �  k  X  E  2    �    B  j  �  �  �  �  �  �  w  B    �  X  �      �    �  �  r  Y  ?  "    �  �  ~  O  &    �  �  �  �  �  {  c  �  �  �  �  �  �  �  �  z  j  ^  V  H  5          �  �  b  �  �  �  �  �  �  �  �  �  �  �  h  N  .  �  �  �  n  =    �  �  �  �  �  �  �  �  �  �  �  �  �  ~  c  G  *      �  �  �  �  �    �  �  �  �  �  �  �  `  6    �  �    z    m  w  �  {  e  H  '    �  �  v  C  
  �  �  �  �  �  �  �  f  Z  O  D  <  4  *       	  �  �  �  �  �  �  �  |  \  <  �  �  �  �  �  �  {  q  e  W  L  G  >  -    �  �  �    V  U  �  $  Z  z  �  �  �  }  e  <  �  �  V  �  ^  �      �  �  �  �  �  �  �  �  �  �  �  �  ~  {  �  �  �  �  �  �    .  N  i  }  �  �  n  W  =  "    �  �  �  P    �  G  �  j  �  �  �  �  �  |  l  [  K  8  %      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    +  S  x  =  *  m  z  r  ^  A    �  �  �  �  e    �  v  ?  M  @  6      �  �  �    �  �  �  �  \  -  �  �  g    �  �  �    2      �  �  �  �  t  S  2    �  �  �  �  g    �  O  �  �  ~  o  `  K  6     	  �  �  �  �  �  �  h  ^  T  J  @  �  �  �  �  i  F    �    "  �  X  �  �  #  �    �    k  ~  b  D  "  �  �  �  �  �  h  R  B  "     �  �  �  n  H  "    �  �  �  �  �  u  [  @  (    �  �  �  �  �  �  �  �  �    �  �  �  �  �  t  W  :    �  �  �  v  E    �  �  m  2  �  �  �  �  o  :  �  �  m    �  w  )  �  �  �  G    �  B  P  <  %    �  �  �  V  $  �  �  �  X  !  �  �  h    �  T  �  �  �  �  |  ]  9    �  �  u  #  �  i  �  �    �  �  j