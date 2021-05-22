CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�� ě��        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       PD�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ȴ9   max       <���        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�=p��
   max       @FǮz�H     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33332    max       @v�z�H     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @�!`            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <�1        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�O�   max       B5 �        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�c�   max       B4�U        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C���        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?.�H   max       C��        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          J        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P4�        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�u�!�R�   max       ?�ě��S�        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ȴ9   max       <���        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�=p��
   max       @FǮz�H     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33332    max       @v~fffff     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @�!`            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q�`   max       ?�ě��S�        W�         
                               A   "      3               "                              I      G         /      	                  ,            '   
            /                  "   $            N>�mOcF;N��O�F�N��N�6N��P�OPM�NJ�GO1��O�<O��O��!O��,PpWP�O���O��O�9�O�IwP4�O�iN2ߎN��O��OgNO�WN�\NL#PD�O��P;�O�3�N��P0 N���O'h�O�-,O��PN�x{N�JO���OmA�N5�{OQ��N��P	�oN�CN�<�O�,cO<u�OFX`O�xN���Np��Or =O���O�<%OB7�N���N�&�N�TN�	[<���<#�
<t�;��
:�o%   �o���
�o�o�49X�49X�49X�D���D���e`B���㼬1��9X��j�ě����ͼ�����`B��`B��h��h���+�+�+�C��C��\)�\)�\)��P��P������w��w��w�0 Ž8Q�@��@��T���q���u��o��t����P���㽛�㽟�w���������������T��j��j�\�ȴ9���������������������������������������������������������������������������������������������������� ))+)"����

�������������������������#(-/<=DNHE<7/#����������������������������������������������������������� )5BN[ejie_[B5")6B[tzzvmfd\ZOB:.(")-68HO[`k������t[B)!-	)BO\fgd^NB6)	����
����������������������������#)16OX_`[OB6)3CNgtw��|utg[NK>3*3(/<=HNUWXVVH<#�)6I[t��������[�������������kmnsz{�����{zmkkkkkkwz{������zxvsruwwww#0<IUX[[_ZI<0
_bn{��������{nb^\Z[_��������������������������������������������������������
<3�����������������������@H@6���������9BNgt��������tg[I769��������������������tz��������������{vot����������������������������������������V_gt����������tg[TPV#0<ITUbibUB@<2!��������������������lmqz������zpmllllll��������������������������������������������������������NRV[gttuv}��ztg[VPNN�������������������������� �����������������������}|������������������������������������������^ainz���������znmha^�����
#(&$
�����#/:<GIJJIJHE<5/+# #pz|�����������}zvqpp��������������������)5BNVTRPMIB5)#gkt������������tgbfg�����������������?IPUbeikjgbYUIFC??>?��
!!# 
������1<@HUW^abaaUH<701111yz�����������zzwyyyy���������������������-�(�'�-�3�:�F�L�O�F�:�4�-�-�-�-�-�-�-�-����������������
�#�/�<�H�Q�H�<�/�#��H�B�<�9�2�<�H�U�^�a�e�n�o�n�d�a�W�U�H�H�ܻû������������ûܻ����������������������
������
�����������������ֺѺɺƺɺֺغ������������ֺֺֺ�¿¾²«²º¾¿��������¿¿¿¿¿¿¿¿�`�M�B�=�:�B�G�T�l�y�����������������m�`�U�H�<�4�/�#���#�/�U�\�a�i�n�xÀ�z�n�U�4�2�4�5�A�M�P�Z�]�Z�M�A�4�4�4�4�4�4�4�4��
���(�-�5�;�=�;�5�5�(��������Z�N�J�A�:�9�A�N�g�s�x�������������s�g�Z�Z�N�5�+�#�"�%�/�A�Z�s�������������s�g�Z���������ʾ׾�����"�.�;�E�@�;��	�𾶿����m�T�L�H�T�`�p�y���������������������S�@�<�?�C�S�x�����������»ȻŻ����x�l�S���|�r�p���������ʾ׾�����׾ʾ������L�J�D�?�B�@�G�Y�^�r�~�����������~�e�Y�L���s�i�g�`�g�h����������������������������������������)�5�N�Q�Q�P�F�B�)��������{�x���������������������������������m�T�C�/�����������	�"�3�;�H�d�y��z�m����������������������%�$����(� �����������(�*�*�(�(�(�(�(�(ùìù��������������������������ùùùù�����������������������������������������N�G�5�,�%�(�5�>�N�^�s�}���������s�g�Z�N�׾ξʾľʾ˾׾����	������	������
��������������
���#�-�#���
�
�
�
�<�2�<�I�O�U�b�i�i�b�U�I�<�<�<�<�<�<�<�<����2�B�hďĿ��������������ĳč�O�B��T�S�K�G�O�T�`�b�m�s�y�z�z�y�w�z�y�m�`�T��Y�<�:�@�r��������������˼���ּ����������������	���#�%�"���	���������������	���	�����������������	�������'�T�a�m�x�������a�H�;��	������*�6�;�:�6�*���������Źűŭũŭ�������������������������Ź�	���������
��"�/�;�B�C�H�L�L�K�C�;�"�	�������������4�M�Q�M�@�'����������ʾ˾ξʾʾ�������������������������������������������������ƳƧƚƁ�w�u�h�g�i�uƄƚ��������������Ƴ����������	���*�6�C�F�M�K�C�6�*����z�u�v�z�������������z�z�z�z�z�z�z�z�z�z��h�\�O�H�N�O�\�h�uƁƎƚƧƦƠƚƎƃ�ù÷ìåêìù����ùùùùùùùùùùù�f�Z���������ּ����!�$�"����㼤���r�f�����������»ûлܻ������ܻлû������L�J�L�Q�Y�d�e�r�~�����~�t�r�e�Y�L�L�L�L����������$�/�=�I�O�S�I�=�0�&��������������������������ʾҾ׾��پ׾ʾ���EED�D�D�D�D�D�D�D�EEE*E+E0E,E*EEEE�E�E�F FFF$F1F=FJFNFJFGF=F9F1F$FFE��ϹιϹ׹ܹ��������	������ܹϹ��#�����#�0�<�E�F�<�0�#�#�#�#�#�#�#�#�U�K�J�K�I�F�H�U�a�n�zÇÒÕÎÇ��n�a�U�����³©¤§²�������
��#�0�5�9�/�#��G�C�:�6�6�@�G�S�`�����������|�y�l�`�S�G�Ľ��ĽȽ׽ݽ������(�+�#������нĿ��ݿۿڿݿ�������
���������������� �����(�(�3�1�+�(������лλû��������ûлܻ޻����ܻллл�ĿĿĿ��ĿľĴĸĿ������������������ĿĿ H f L 9 q O l F D 6 2 /  D H % = : 4 + =  % m � C 0 U \ l z ; y U B D 9 g " @ 4 . ] 2 7 G / � N X [  B T g @ / l $ R ) * U Q  e  !  �  z  W  �  y  �  �  \  �      P  4  g  �  h  _  �  �  �  H  �    �  �  |  �  �  c  D  �  �  u  �  �  �  K  A  	  �  �  �  V  �  8  �  $  �  n  �  �  g  �  �  �  �  (  �  �    �  �<�1���
%   ������t��e`B�e`B��P���e`B���o�����8Q������<j�\)�@��49X�m�h�P�`�T�����\)�49X�#�
�<j��w��w����#�
��������,1���T�'<j�Y��m�h�P�`�H�9�����-�@���7L�aG���Q콍O߽���� Ž�Q�������1���T��/�����l���h����"ѽ���B,�xB�B!��B �DB ɩB�?Bt�B+ISB(�BU�B��B��B�5B�CB*�BƠB#,[B!ԗB��B�9B~�B�DB��A�O�A��^B%��B(��BP�B2B�9B�B�CBA�B	�VBՄB��BvBkB	�^B%�hB5 �A���B�lBcB]�B	GRB��B-7XB)��B��B.IBe�B�XB{�B��B�B2�B
E�B_*B&��BCBK�B�BƟB,�kB�B!;0B gB AB�wBA^B+?�B8�BFB�SB=�BìBU�B0BB B#�xB!�4B��BǎB=�B>kB��A��7A�c�B&="B(�WB<B�bB��B�(B&rB��B	�BBB3-B@RB	��B%��B4�UA���B�JB��B�=B	:�B�PB-�5B)��B�;B��B��B��B@B�8B��B�qB
��B�~B'G:BÏBC�B��B�@{�lA�w�A�H4@�>A�T�@D�1A�� AjN�A�H�A;�A��cA�3�A�z�AW�hAp��@�5�AOz_?�QIA�7�A���A�dhA�g=A��HA�sA���A���A�X�AUZ�A�nBA��A޸�Ai��@�V(AY�KAX��A��fA���A�-sA���@�AM�B��B��A���A�(<B�pA�6�@��@�\?��NB
AN#SC�\C���?��A�S�A��A��Ag�A.�oA�nZA�O�@�g�A���@|B!A���AĚ<@�A�d@B��A�u�Ai�A�zYA;@,A���A���A�y~AVýAo1/@��AI5�?�>A�o�A��A�bA�|�A���A� A��A��A��AT
A�yKA�4A�IAj��@�aAZK�AX��A�~�A�HA�^eA�:7@�A'AL��B��B�-A�|�A�:gB,~A�v�A�q@��?��NB	�JAM�C�XoC��?.�HA띴A�}?A��kA�PA/�0A�c�A�~+@���A��         
                !               A   "      4               #                              J      G         /      
                  -         	   '               /      	            #   $                        #            +               #   '   '   %   )            !   =            !               7      7         )            #         !               5                              %                              !                              '   '   #   )               =            !               !      5         %                                    1                              %                  N>�mO!�~N��O���N��Nr(gN��O8�BN|9NJ�GOE/O�<O�@PO��!O��,O�kP�O���O���O�9�O�6�P4�O�iN2ߎN��O��OgNN�KuN�\NL#O���O��P+>�O4-N��O��.N���O'h�O�-,N�/�N�x{N�JO�uOC��N5�{OQ��N��PN�CN�<�O�,cO��OFX`Of�N���Np��Oe�O���O�<%OB7�N���N�&�N���N�	[  U  %  u  U  I    �  �  G  �  �  �  u  ~  �  �  U  �  �  �  X    �  V    �  �  �  �  `  	�  �  	Q  b  �    G  �  �  M  (  �  �  �  �    �  i    �  q  �  
<  }  >  �    k  `  7  V  f  �  <���;�`B<t�;�o:�o��o�o���㼛��o�T���49X�o�D���D�����㼛�㼬1���ͼ�j�������ͼ�����`B��`B��h��h�\)�+�+�e`B�C����@��\)�0 Ž�P��P���D����w��w�#�
�D���8Q�@��@��Y��q���u��o���P���P���-���㽟�w���
�����������T��j��j�ě��ȴ9����������������������������������������������������������������������������������������������������()*)!����

������������������������������#//0<<><5/# �����������������������������������������������������������&)5BNS\aba_[NB5(  "&)6B[tzzvmfd\ZOB:.(")-68HO[`k������t[B)!-)BOYacc`RHB)����
����������������������������)6=OS[\XOB6)3CNgtw��|utg[NK>3*3#*/<AHMTVVTM<#�)6I[t��������[�������������kmnsz{�����{zmkkkkkkwz{������zxvsruwwww#0<IUX[[_ZI<0
_bn{��������{nb^\Z[_�������������������������������������������������������
 #�����������������������=A8���������NNU[gt�������tg`[QNN��������������������wz���������������zuw����������������������������������������V_gt����������tg[TPV #0<AGE><0-#!      ��������������������lmqz������zpmllllll���������������������������������������������������������NRV[gttuv}��ztg[VPNN�������������������������������������������������}|������������������������������������������_aknz��������zunkda_�����
#(&$
����� #$/8<EHIJIH<6/,##  pz|�����������}zvqpp��������������������)5BNQSRPLHB5)$gkt������������tgbfg�����������������?IPUbeikjgbYUIFC??>?��
!!# 
������1<@HUW^abaaUH<701111zz�����������zzwzzzz���������������������-�(�'�-�3�:�F�L�O�F�:�4�-�-�-�-�-�-�-�-����������
��#�.�/�;�C�H�J�H�<�/�#��H�B�<�9�2�<�H�U�^�a�e�n�o�n�d�a�W�U�H�H��ܻлû����������ûܻ���������������������
������
�����������������ֺҺɺǺɺֺ޺������� �����ֺֺֺ�¿¾²«²º¾¿��������¿¿¿¿¿¿¿¿�`�T�N�H�E�G�J�T�`�m�r�������������y�m�`�H�@�<�;�<�H�H�U�Y�a�g�h�a�U�H�H�H�H�H�H�4�2�4�5�A�M�P�Z�]�Z�M�A�4�4�4�4�4�4�4�4�����������(�5�9�<�:�5�2�(���Z�N�J�A�:�9�A�N�g�s�x�������������s�g�Z�A�<�2�.�0�5�A�N�Z�g�s�|������s�g�Z�N�A���������ʾ׾�����"�.�;�E�@�;��	�𾶿����m�T�L�H�T�`�p�y���������������������x�l�S�G�B�B�F�S�l�x�����������û������x���|�r�p���������ʾ׾�����׾ʾ������L�J�D�?�B�@�G�Y�^�r�~�����������~�e�Y�L�s�m�g�k�s�����������������������������s��������������)�5�N�Q�Q�P�F�B�)��������������������������������������������m�T�C�/�����������	�"�3�;�H�d�y��z�m����������������������%�$����(� �����������(�*�*�(�(�(�(�(�(ùìù��������������������������ùùùù�����������������������������������������N�G�5�,�%�(�5�>�N�^�s�}���������s�g�Z�N�׾վʾʾʾ־׾������������׾׾׾��
��������������
���#�-�#���
�
�
�
�<�2�<�I�O�U�b�i�i�b�U�I�<�<�<�<�<�<�<�<�h�[�P�T�[�f�tĀĦĳĿ������ĿĳĚā�t�h�T�S�K�G�O�T�`�b�m�s�y�z�z�y�w�z�y�m�`�T����Y�F�=�;�B�r����������ʼ��ּʼ�����������������	��������	����������������	���	�������������������	�	���/�H�T�a�m�v�|�~�|�m�T�;�"�������*�6�;�:�6�*���������Źűŭũŭ�������������������������Ź�	���������
��"�/�;�B�C�H�L�L�K�C�;�"�	�����������'�*�+�'����������������ʾ˾ξʾʾ�������������������������������������������������Ɓ�y�u�m�k�uƇƚ����������������ƳƧƚƁ������� �����*�6�D�J�H�C�8�*����z�u�v�z�������������z�z�z�z�z�z�z�z�z�z��h�\�O�H�N�O�\�h�uƁƎƚƧƦƠƚƎƃ�ù÷ìåêìù����ùùùùùùùùùùù�r�f�y�������ּ����!�$�!�����㼤���r�����������»ûлܻ������ܻлû������L�J�L�Q�Y�d�e�r�~�����~�t�r�e�Y�L�L�L�L����������$�/�=�I�O�S�I�=�0�&��������������������������̾׾׾ھ׾оʾ�����EED�D�D�D�D�D�D�D�EEE*E+E0E,E*EEEFE�E�E�E�FFFFF$F1F=FFF=F8F1F$F#FF�ϹιϹ׹ܹ��������	������ܹϹ��#�����#�0�<�E�F�<�0�#�#�#�#�#�#�#�#�U�K�K�K�J�H�H�U�a�n�zÇÑÔÍÇ��n�a�U�����³©¤§²�������
��#�0�5�9�/�#��G�C�:�6�6�@�G�S�`�����������|�y�l�`�S�G�Ľ��ĽȽ׽ݽ������(�+�#������нĿ��ݿۿڿݿ�������
���������������� �����(�(�3�1�+�(������лλû��������ûлܻݻ����ܻллл�ĿĿĿ��ĿľĴĸĿ������������������ĿĿ H O L 9 q J l 8 $ 6 ' /  D H ( = : 3 + 3  % m � C 0 H \ l Z ; s B B B 9 g "   4 . V / 7 G / u N X [ # B A g @ , l $ R ) * T Q  e  �  �  _  W  �  y  �  {  \  L      P  4    �  h    �  r  �  H  �    �  �  �  �  �  �  D  �  a  u  <  �  �  K  �  	  �  w  �  V  �  8  E  $  �  n  O  �  '  �  �  �  �  (  �  �    �  �  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  U  V  W  X  Z  Y  T  O  J  E  =  0  $       �   �   �   �   �      �  �  $      	  �  �  �  �  �  {  9  �  �  5  �  �  u  o  j  e  _  J  1    �  �  �  �  �    �  �  �  �  �  �  B  T  P  E  3  '  K  P  C  /    �  �  �  "  �  V  #  �  �  I      &  =  -  �  I  E  >  6  -  "      �  �  �  �  �          �  �  �  �  �  v  C    �  �  k  0  �  �  �  u  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  N  4      �  R  Y  Y  n  �  �  �  �  �  �  �  �  �  v  /  �  �    �   �  �  �  �      &  .  7  A  F  C  7  $  
  �  �  �  v  W  5  �  �  �  �  �  �  �  ~  u  m  e  ]  T  L  C  9  1  )  !    �  �  �  �  �  �  �  �  �  p  U  2    �  �  I    �  y  �  �  �  �  �  �  �  u  [  <    �  �  �  G    �  n    �  h  M  �  	  ;  \  p  u  q  c  H    �  �    �    j  �  |  �  ~  o  Y  ;  J  @  (    �  �  �  �  �  `  1  �  �  L  �   �  �  �  �  �  �  �  �  {  t  m  e  ]  Q  =     �  �  �  m   �  I  �  �  �  �  �  �  h  K  4  @  R  F  $  �  �  [  �    �  U  B  �  �  b    �  �  r  =    �  �  �  �  g  -  �  (   �  �  �  �  �  �  �  �  �  �  q  Z  B  %  
  �  �  �  �  �  f  �  �  �  �  �  �  �  �  �  �  �  �  d  5  �  �  �  4  �   �  �  �  �  �  �  �  �  �  �  o  N  *    �  �  �  �  ]    �  0  U  J  .    �  �  �  q  J    �  �  h  #  �  l    �  b    �  �  �  �  \  b  p  \  6    �  �  t  4  �  �  �  :   �  �  �  �  �  �  �  �  �  �  h  2  �  �  `    �  S  �  �  �  V  Q  L  G  B  =  8  (    �  �  �  �  �  �  �  r  _  L  8      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  i  P  5      �  �  �  �  �  �  n  E     �   �  �  �  |  l  \  M  >  0  "      �  �  �  �  �  l  N  -    �  -  W  m  |    y  j  T  6    �  �  �  ]  *  �  �  �  F  �  �  �    q  b  S  ?  )      @  f  {  l  \  M  ;  (    `  Q  B  3  #          
    (  ;  I  N  S  Y  ^  c  h  ]  �  �  �  	  	}  	�  	�  	�  	�  	_  	.  �  �  !  �  �  �  ,  �  �  �  |  n  [  G  3      �  �  �  �  �  �  �  {  �  �  �  	G  	P  	F  	)  �  �  [  �  �    -  k  )    �  >  �  �  �    �  �    6  H  R  Z  a  `  W  F  -    �  x  )  �  ~  )  �  �  �  �  �  �  �  }  u  m  d  [  R  H  >  4  (      �  �  �            �  �  �  �  m  4  �  �  i    �  V  �  !  G  @  9  2  +  #        �  �  �  �  �  �  �  w  V  5    �  �  �  s  V  ,  �  �  �  �  �  �  z  ^  8    �  �  �  �  �  �  �  �  �    e  I  -    �  �  �  w  R  -  	  �  �  �     2  (        ,  ?  I  L  L  B  .    �  �  �  k  o  `  (        �  �  �  �  �  �  n  Q  4    �  �  �  ]  &   �  �  �  u  c  P  =  (    �  �  �  �  �  r  P  -    �  �  �  |  �  �  �  l  N  /    �  �  �  j  6     �  �  y  c  Y  Z  �  �  �  �  �  �  �  �  X    �  �  $  �  (  �  �  "  0  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  ~  g  j  x  a  F  .    �  �  �  z    g  �  r  Z  ?  "    �  �  �  �  n  M  ,  	  �  �  �  q  H    <  \  @  Q  )  �  �  �  U      �  �  W    �  P  �  %  U    �  �  �  �  �  �  }  c  G  *    �  �  �  �  a  '   �   �  �  �  �  �  �  �  t  k  G    �  �  w  B    �  �  i  P  R  q  S  4    �  �  �  q  @    �  �  m  4  �  �  ^  	  �    �  �  �  �  �  �  �  �  x  e  S  @  )    �  �  �  |  G    
<  
  	�  	�  	�  	g  	&  �  �  m  A    �  �  7  n  V    �  D  	  v  `  D  (    �  �  �  m  "  �  y    �     �    7  D  >  =  =  4  '    	  �  �  �  �  �  �  p  @    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  r  g        �  �  �  �  w  M    �  �  Y    �  >  �  )  �  p  k  D    �  �  �  n  :    �  �  p  I    �  �  8  �  q  �  `  V  I  6       �  �  �  v  B    �  e    �  �    n  �  7  '      �  �  �  �  �  �  p  D    �    u  �    E  <  V  K  ?  /      �  �  �  m  =    �  �  �  �  }  s  Z  <  f  _  W  L  ?  -       �  �  �  y  P  &  �  �  �  �  �  �  �  �  �  �  e  .  �  �  X    �  �  T    �  u  �  *  _  �      	    �  �  �  �  �  �  �  �  �  �  �  �  �  s  d  U