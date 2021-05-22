CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�333333       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�}�   max       P��C       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <D��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @Ffffffg     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vyp��
>     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O�           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�J@           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �V   max       <o       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0�?       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B0J�       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =d��   max       C���       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >9��   max       C���       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          L       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�}�   max       P�@D       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���ߤ@   max       ?�ۋ�q�       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <D��       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @F:�G�{     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vyG�z�     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @O�           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @� �           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�֡a��f     �  Y|               K                  B      
         1            &   '      
   
   
               B   +               0            	      	   %                     	                  I      (                        -N<�PN�ӤNbmN9ΠP��COs+Nk��O
J�PNO.��Pa�aOX��Nв�Od�qOM�QO��IO��pO��P��PSJO� !OF�DOHҜN`]�Njh�O�;N��sP�PO&G�PP��P�G�O"żO��N���O��oO�B�N��N���N���N��#N��cN7Q�O��xOE�O$KN��M�}�N���N�w�O-CN���O��N�n�O��O�}P+̝NI/�O2��N*�Ng��N'c�N��N��N��N��]O��<D��;�`B%   ��o�D����`B��`B��`B�D���D����o��t����㼬1��1��1��1��1��j�ě��ě��ě�����������/��`B��h��h�����������+�+�C��C��\)�\)�t��t���w�#�
�#�
�'0 Ž0 Ž8Q�<j�H�9�H�9�P�`�T���aG��ixսm�h�m�h�q���u�u��C���C����w���
�� Ž��
#'($#
`adhmvz��ztma^``````
 

	��������������������#0b�����rmUL<0*/��������������")5<;5.)2<>HU[Z\ZUTHA<933322�
/;HQT[[TH;"	����Z[^gt���������vtia[ZL[t����������g[OJLKL�������	������������������������������\alz��������zpma[VU\��������������������������������������������������������������6CO\w�rO6*���������������������z������� ��������S[]hko{������h[RJHJSt~���������������tit7;HTakljgaYTHD@;7/07T[_gmpmg\[ZQTTTTTTTT����������������������������������������25=BN[\[XRNB53222222#0Ibn{�|slUI<0'��������������������������&5���������5t����������g[RVUO95�
#07/)/<HMH/#
��v�����������������v������������������������"��������*6BCDDFE>)������������������������O[gtz����tgd[WOOOOOO367BCO[hiha[[OB63333 '	���    05:BGJNSUNB95)000000��������������������|����������������~z|)68BFIJB64)t{������������vtqppt����������������������������������������z{����������{ytszzzz��������������������KNY[gt��������g[RNJK������������������������������������������������������~{����nz������������~zunnn���������������������������!$# #*/<A<??>:/*#��������������������-04<>AA<0*&'--------����
������������),36660)@BDLKMN]gt|tllg[NKB@rt�����������|trrrrr[[\hopstwtih[ZVX[[[[
%/<HKTTVTH<#�������������������������6�3�)����!�)�6�;�B�J�J�B�6�6�6�6�6�6�׾Ҿо׾�������׾׾׾׾׾׾׾׾׾��U�O�L�Q�U�_�a�c�f�f�a�^�U�U�U�U�U�U�U�U�����|����������(�A�Z�s���{�f�4��н�����������������������������������������m�m�a�W�U�a�m�u�z�}�|�z�m�m�m�m�m�m�m�m������������������������������������������������������	�����2�9�9�1�"�	�����������������������ĿѿڿڿѿοĿ������s�i�m�|������������%��������������sìåàÕÏËÊÓàäìôù��������ýùì�U�L�I�<�8�4�<�I�U�b�h�n�t�t�n�b�U�U�U�UƳƭƧƜƠƧƳ������������������������Ƴ��������������$�0�9�=�>�<�4�0�$���Y�M�@�'����'�4�M�f�t�����������r�Y�m�T�H�A�<�:�<�G�T�]�y���������������y�m�;�8�"���ܾԾ־����	��$�/�1�5�D�H�G�;��޿׿ٿ����5�A�P�]�d�d�T�5�(�����T�G�;�/�"�;�G�`�m�y�������������y�m�`�T�	������������%�/�D�H�J�Y�^�Z�T�H�;�"�	�޾׾־̾Ⱦƾ¾ʾ׾���������� ���������������(�5�C�N�T�U�N�A�:�5�(�����������$�)�*�)��������������������������������������������������������������������������������������V�O�I�G�A�D�I�L�V�b�f�i�e�b�V�V�V�V�V�V���}�s�g�`�`�e�s�������������������������/�#���#�)�/�<�H�L�U�Z�a�i�a�_�U�H�<�/�B�;�6�'�'�,�@�O�tĦĶ����Ŀĳčā�h�O�BĦĕėĦĿ�����#�<�P�V�J�M�J�0�
����ĳĦ¼¼±¦¨²����������������������¼�������սѽݽ������"�$�!����������"�*�-�6�8�C�H�C�6�*�����������������������*�:�A�B�>�6�.�*���x�t�k�d�_�W�Y�l�x�����������»��������x�������������ɺֺ�����ֺֺɺ�����������������	����	���������������������������������������������������ʾľ������Ǿʾξ׾�������׾ʾʾʾʾ��������������ʾ׾��׾Ҿʾ�����������ÓÒÎÓàìöìçàÓÓÓÓÓÓÓÓÓÓ�~�e�L�=�(�'�6�@�Y�e�r�����������������~�s�m�g�o�s�����������������������������s®¦¡¦§©²¿��¿����������������¿®�m�m�b�a�^�a�j�m�s�z���������������z�m�m�S�P�F�:�9�:�E�F�J�S�X�U�S�S�S�S�S�S�S�S�����'�4�5�@�F�M�M�M�A�@�4�'�������������
��!�#�(�#��
������������������ ��������������������������������������������������������������������������������������&�*�-�/�*������ŹŶŶŹ���������������������������ƾ���������������������������������������������ü������)�$������ּʼ������Y�M�D�I�Y���������ּ߼ּѼ�ټ�����f�Y����߼�����
�������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�FFFE�FFFFF$F'F'F$FFFFFFFF����������������������������������	�������������������������½ĽŽнٽѽнĽ�����������Ç�}�z�n�a�U�T�H�A�E�H�N�U�a�r�ÄÇÊÇÓÒÌÇÇÄÇÓàìîóíìãàÓÓÓÓ�����������������ùƹϹййϹù���������EE
EEE&E(E"E*E7ECEPE\EfEbEbE\EPECE*E F J 5  M b J 9 R 5 V I X J * M O  k - P W 3 H r N E > * > G f Q L * 1 @ p [ K D N B 4 J G p F : f c 9  G R 6 j d T F Z # s 8 P L  j  �  +  �  "    k  :  �  �  �  �  �    �  x  �  �  �  R  �  �  �  n  �  N  �  �  h  �  W  �  n  �  %  �      �  �  �  N  �  �  �  �  b  �  �  �  �  C  �  9  �    o  �  =  �  j  �  ~  �  �  `<o��o�o�49X���w��1�D����t��C���j���T�@���`B��h�0 Ž�\)�<j�8Q�<j�y�#�}���\)�t���P��w��P�L�ͽe`B�Ƨ𽗍P�8Q�q����P�}󶽥�T�#�
�'�w�8Q�#�
�@����-��+�Y��@��<j�H�9��%�ixս]/����m�h�����-�$ݽ�o������t���o��t�������置{�����VBMcA��,B$� B5gB&�CBEB��B0?A��B	�B
j�B.B��A�uB��B �ZB+EAB0�?B�~B3�B��B��A��B	B�B��B��B&�PBm�Be=B
4pB�0B �Bl�B��B��B!'�B	Z�B�wB��B��B��B�B8�B
N�B�%BB)$�B�B	��B^�B{�B
�B�B-qNBUCB�\B�HB:�B%�B_-BB��B
��B�SB��BT�A�z�B$��B-RB&�lB�rB�B?�A�~�B
DB
|�B��B��A�|�B¥B őB*�oB0J�BH�B(AB��BZ�A�o�B˩BM�B��B��B&��BC�B�~B	�VB~�B D�B,�B��B��B!��B	�B��B@BŭB�<B@8B@�B
D"BƼB@B)>IB��B	V�B�YB~MB&5BũB-�bB�$B�BB��B@B%��BF�B?�B�	B
��B�CBD�A�d�A�,�ATλA��WA+�A�Q�A�K�A��6A��JAv��A�V�A�,UA��BL�B	v�@�Ak�fAZ��A��lAkA�_�AT�A��*A���A���A�&B��A�9Aù:AܴsA���A�& A/�;A�h�A��}@���@6خAXЪA���ASLpAPӾA˜�?�"�A�%�A�IfA�i�@���@�2�A���A�٤A��%A�[A�RuAI��Aw@�rA.}C�fXC���@��A0�A&C�A�9A��=d��C��EA��	A�3�AT��A�n�A,��A��>A�y%A��3A�oLAw�wA�~ ÄA��BOB	�f@�B3Am	�AZ5pA���Aj��A�vqAU
GA�]�A�|rA�~�A��B�VA�rAÁA�b�A�dA���A0�A���A��A@�C�@<5�AY�A�LHAS�IAQHA�gG@SA��ZA���A�d�@��@�{A�o�A�:A��0A��|A���AI��A0\@��1A�C�jfC���@��A1GA%cA�E�A˪�>9��C���      	         L                  B      
   	      2            &   '      
            	         C   ,               0            
      	   &                     	                  J      )                        .               C            )      3                  %   +   -   %   !                     )      /   7            !                        !                                    #   +                                             )            '      1                        +                           )      -   3                                    !                                    #   )                              N<�PN�ӤNbmN9ΠO�kOW��Nk��O
J�P L�O.��P7	�O(J~N�e�Od�qO0��N��O���O%S�O���O�<*O9�lOF�DOHҜN`]�Njh�N��N��sP�POϡP>�WP�@DO"żOF�fN���O�y�O�ڣN��N���N���N�'�N��cN7Q�O��xO:a�N�SZN��M�}�N���NYȣO-CN���N�w�N���O��O�}P �4NI/�O�tN��Ng��N'c�N��N��N��N��]OFCC  (  �      �  }  �    8  Q      �  �  _  :  �  p  q  1    U  �  �  �    �  >  �  "  U  �  �  %  �    %  �    U    �  �  V  �  �  �  �  �  H  �  -  7  �  8  �    
J  �    c  �  {  J  �  	#<D��;�`B%   ��o��P�o��`B��`B�e`B�D�����ͼ�j��1��1��j�'ě����ě��C��C��ě�����������/��h��h��h�+�\)�o����P�+��P�,1�C��\)�\)��P�t���w�#�
�'8Q�0 Ž0 Ž8Q�L�ͽH�9�H�9�aG��Y��aG��ixսy�#�m�h��%�y�#�u��C���C����w���
�� Ž���
#'($#
`adhmvz��ztma^``````
 

	��������������������#)0<IUhqwvqjYUI</(##��������������")5<;5.)2<>HU[Z\ZUTHA<933322��/;HOQUTC;"	����Z[^gt���������vtia[ZNPR[t�����������gTNN���������������������������������������\alz��������zpma[VU\������������������������������������������������������������%*/6<COY\^]QC6*����������������������������������NSY[hntv{~��th[QOMMNt~���������������tit7;HTakljgaYTHD@;7/07T[_gmpmg\[ZQTTTTTTTT����������������������������������������25=BN[\[XRNB53222222#0Ibn{�|slUI<0'�������������������������(-)���������:Nt�����������[RWVO:�
#07/)/<HMH/#
�������������������������������������������������������+6=?A?6)������������������������O[gtz����tgd[WOOOOOO367BCO[hiha[[OB63333���05:BGJNSUNB95)000000��������������������|����������������~z|)68BFHDB63)rtw���������|tsrrrrr����������������������������������������z{����������{ytszzzz��������������������KNY[gt��������g[RNJK����������������������������������������|���������������||||nz������������~zunnn�����������������������������!$#!#,/7<=<82/'#��������������������-04<>AA<0*&'--------����
������������),36660)@BDLKMN]gt|tllg[NKB@rt�����������|trrrrr[[\hopstwtih[ZVX[[[[
#+/<HNPH</#

�������������������������6�3�)����!�)�6�;�B�J�J�B�6�6�6�6�6�6�׾Ҿо׾�������׾׾׾׾׾׾׾׾׾��U�O�L�Q�U�_�a�c�f�f�a�^�U�U�U�U�U�U�U�U���������������Ľнݽ���(�.�6�*��н��������������������������������������������m�m�a�W�U�a�m�u�z�}�|�z�m�m�m�m�m�m�m�m��������������������������������������	����������������������/�7�7�/�"��	���������������������ĿѿڿڿѿοĿ������������q�w�������������	������������ìàØÓÒÑÏÏÓàìðùü��������ùì�U�R�I�>�E�I�U�b�c�n�q�q�n�b�U�U�U�U�U�UƳƭƧƜƠƧƳ������������������������Ƴ���������
���$�0�7�=�=�=�:�2�0�$���Y�S�M�@�4�3�0�4�@�L�M�Y�f�r�t�}�w�r�f�Y�m�T�K�D�?�>�A�G�T�`�y���������������y�m��	��������߾������	���$�%�"�!�����߿ٿݿ����5�A�N�\�c�c�Q�(������`�T�G�=�8�C�G�V�`�m�y�������������y�m�`�"��	����	���"�/�;�H�S�U�S�H�B�;�/�"�޾׾־̾Ⱦƾ¾ʾ׾���������� ���������������(�5�C�N�T�U�N�A�:�5�(�����������$�)�*�)��������������������������������������������������������������������������������������V�O�I�G�A�D�I�L�V�b�f�i�e�b�V�V�V�V�V�V���}�s�g�`�`�e�s�������������������������/�/�#� � �#�,�/�<�H�I�U�V�a�W�U�H�<�/�/�B�9�+�+�/�D�O�[āĚĦ������ĿĦā�h�O�BĳğĕĘĦĿ�������#�<�N�U�I�I�0�
����ĳ¼¼±¦¨²����������������������¼��������߽۽۽ݽ�����������������"�*�-�6�8�C�H�C�6�*����������������������*�5�<�=�9�2�*����������z�p�i�l�x���������������������������������������ɺֺ�����ֺֺɺ�����������������	����	���������������������������������������������������ʾƾ������ʾ׾�������׾ʾʾʾʾʾʾ��������������ʾ׾��׾Ҿʾ�����������ÓÒÎÓàìöìçàÓÓÓÓÓÓÓÓÓÓ�~�e�L�=�(�'�6�@�Y�e�r�����������������~�s�n�h�p�s�����������������������������s¿¶²§±²¿������������������¿¿¿¿�m�m�b�a�^�a�j�m�s�z���������������z�m�m�S�P�F�:�9�:�E�F�J�S�X�U�S�S�S�S�S�S�S�S�����'�4�5�@�F�M�M�M�A�@�4�'�������������
���#�%�#��
������������������ �����������������������������������������������������������������������������������!�)�'�����������ŹŷŹŹ�������������������������Ҿ���������������������������������������������ü������)�$������ּʼ������f�Y�M�G�E�K�Y�������ʼѼϼڼ޼ؼ�����f����߼�����
�������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�FFFFFFFFF$F&F%F$FFFFFFFF����������������������������������	�������������������������½ĽŽнٽѽнĽ�����������Ç�}�z�n�a�U�T�H�A�E�H�N�U�a�r�ÄÇÊÇÓÒÌÇÇÄÇÓàìîóíìãàÓÓÓÓ�����������������ùƹϹййϹù���������EEEE$E*E0E4E5ECEPE]E\E\E^EXEPEEE7E*E F J 5  C ` J 9 R 5 Q O R J $ B Q I g " 2 W 3 H r @ E > + : G f F L ' $ @ p [ N D N B 2 M G p F 7 f c :   G R 1 j P K F Z # s 8 P V  j  �  +  �  y  �  k  :  �  �    |  �    o     �  �  �  b  �  �  �  n  �  &  �  �  !  S    �  �  �  �        �  �  �  N  �  �  �  �  b  �  i  �  �  �  �  9  �  �  o  S  "  �  j  �  ~  �  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  (  "            �  �  �  �  �  �  �  �  �  a  3     �  �  �  �  �  �  �  �  �  �  k  I  %    �  �  �  v  P  "  �    t  j  _  T  J  ?  5  *            �   �   �   �   �   �   �        �  �  �  �  �  �  O    �  �  h    �  v  :    �    =  J  @  _  k  l    �  �  �  ^    �  �  '  �  R  X   �  {  }  |  y  t  q  s  x  v  o  d  T  ;    �  �  �  ]  9  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  �  �  �  q  F    �  �    Z  =  0  5  8  6  0  &    �  �  �  �  ^  $  �  �  b  -     �  0  Q  E  8  (      �  �  �  �  �  �  �  s  S  /  
  �  �  �  �  �  �      �  �  �  �  r  /  �  �  6  �  v  �  H  k  e  �  �        �  �  �    M    �  �  �  �  :  �  e  �  Z  �  �  �  �  �  �  �  �  �  �  s  c  O  :  !    �  �  �  ^  �  �  {  l  \  L  =  .    �  �  �  �  �  z  a  H  *     �  E  V  ^  Z  U  N  D  7  &    �  �  �  a    �  m    �  �  H  r  �  �  �    %  6  9  0    �  �  �  K    �  |  �  �  �  �  �  �  �  �  �  �  �  �  x  `  C    �  �  O  $  �  l  Q  I  E  I  n  n  p  p  n  d  X  I  8  !  �  �  �  P  �  >  V  q  k  ]  J  1    �  �  �  V  I  <  %  �  �  a  1  �  c  �  �    !  ,  1  -      �  �  �  y  H  
  �  Q  �  &  �  M  �  �  �  	      �  �  �  �  `  2  �  �    �  �  �   �  U  J  >  3  (      �  �  �  �  ~  `  Q  7  	  �  �  <   �  �  �  �  �  �  �  �  t  _  J  5       �  �  �  p  ;  +    �  �  �  �  �  �  �  �  �  z  h  U  @  )    �  �    �  �  �  �  �  �  �  �  )  *  *  %      
  �  �  �  �  �  �  Y      	  �  �  �  �  �  �  t  X  <  !    �  �  �  a    �  �  �  �  �  �  �  �  t  V  7    �  �  �  �  U  )  �  �  �  >  %  
  �  �  �  �  y  ]  D  2  "    �  �  �  �  �  a    �  �  �  �  �  �  �  �  �  n  C    �  �  �  a  .  �  �  �  �        �  �  �  [  6    �  �  �  ]    �    d  (  t  Q  7     �  �  x  D  �  �  �  �  �  b    �  �  `  �  �   �  �  �  w  d  C  !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  V  +  �  �  [  �  w  �  8  %        �  �  �  �  �  �  �  �  �  z  g  Q  ;  %     �  �  �  �  �  �  �  �  �  {  V  -    �  �  c    �  q  
  O  �  �         �  �  �  �  b    �  �  b    �  g  �  X  �  %           �  �  �  �  �  �  �  �  v  e  T  :  �  �  k  �  �  �  �  �  �  �  �  �  o  Y  B  *    �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  ~  t  i  _  [  Z  Y  W  V  *  ?  S  I  ;  +      �  �  �  �  �  �  m  L  (      �    y  t  o  j  e  b  _  \  Y  Y  Z  \  ]  _  W  M  B  7  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  }  s  d  S  A  /    �  �  �  �  �  {  I  G  G    �  �  �  �  �  k  (  �  �  X  M  Q  :    �  �  �  �  U  "  �  �  }  =  �  �  Z  �  <  �  j  �  �  �  �  �  �  �  �  �  �  k  E    �  �  ;  �  o    �  �  �  �    {  w  s  o  k  e  ^  W  P  I  ;  ,       �  �  �    
          !  "  #  %  &  "      �  �  �  �  �  �    w  o  g  `  Y  S  L  ?  +       �   �   �   �   q   Q  �  �  �  �  �  �  �  �  �  �  x  c  L  2      �  �  |  ?  H  D  ?  8  /  %      �  �  �  �  �  �  �  s  _  O  D  8  �  �  �  ~  m  Z  H  5  %      �  �  �  �  �  �  �  �  �  �  �      &  +  +  &      �  �  �  ^  -  �  �  �  K    .  0  3  6  3  .  *        �  �  �  �  �  w  [  B  )    �  �  ~  f  N  5      �  �  �  �  w  f  V  H  :  -      8  -  #  %      �  �  �  �  �  X    �  �  \    �  :  �  �  �  �  �  P  /  	  �  �  �  �  G  �  �    k  �     �  �                  �  �  �  �  �  w  Q  +    �  �  I  	�  	�  
G  
#  
  	�  	�  	p  	>  	  �  �  S  �    M  �  �  �  �  �  �  �    �  �  �  `  !  �  �  }  T  /    �  �  �  �  �      
      �  �  �  �  �  �  �  �  �  �  
    /  B  T  c  Z  Q  H  ?  5  *      	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  j  W  D  )     �   �  {  l  ]  N  ?  3  '                            J  <  .         �  �  �  �  �  �    l  Z  G  '    �  �  �  �  �  �  e  G  ,      �  �  �  �  r  F    �  �  U    �  �  �  	  	"  	
  �  �  <  �  �  %  $  �  �    �    j  