CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��vȴ9X       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       PƼ�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �J   max       =+       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?B�\(��   max       @E�z�H       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @v�z�G�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O�           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       <�h       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��G   max       B-C�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��n   max       B-��       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?%:�   max       C��V       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C��       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          }       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�=�       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���3��   max       ?׎�Mj       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �J   max       =+       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?O\(�   max       @E�G�z�       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v�z�G�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @O@           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         DX   max         DX       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?׋�q��     0  ^                  <   |   >      
               D         	   G                  	               4                                    +   b            B      
   
      
      -      	               E   )         
                  )No}=N�ΧNf2�N�:�N(�UP�=�P���P���O'�~N��:N�_5Ot�IN�Nbw�PƼ�O~��M�r�Nr��P`HOv�6OK�N���O��O;�oN��OޗrO%��N�FN�~O�D�OiO�D�N���O^��N��hOh �O��O59�OYNI�N�N�P�MPF�N�\�N#ߒN�>P*;M�#�N�DN�N�`NވN�? O�o�N��N�m�Oa��O-ХNlN��BP%��O��/O)߆Ov<'N���OH=;OD�Nu��N�K-M��ObR=+;�`B;�o;o:�o��`B�t��t��#�
�T���T���T����o��C���t����㼛�㼴9X�ě����ͼ�/��`B��h�������o�o�+�t���P��P��P����w�#�
�#�
�0 Ž0 Ž0 Ž0 Ž0 Ž8Q�H�9�H�9�H�9�L�ͽL�ͽL�ͽP�`�T���]/�e`B�m�h�u�}�}󶽁%��%��o��o��o��o������������������{�� žJ##022200#########!#'/<HUacaWUH</)'#!!knz��������zwnkkkkkkLN[gtxtqg\[ZPNLLLLLL���������������������
#0<{������U<#��������
0Ukw~zjUI0���������������������}~�u{��������������zuuu>BNN[^ghgf[ZNEB:>>>>##/<<CB=<//#!������)2.!������JO[_c]hjh[ONFHJJJJJJ�
!"��������BN[��������[5������������������������������������w����������{wwwwwwww%6BTZ[_hpx{���tVB)"%v����������������}}v(-5BFNQ[\Z[\[N?50)((�����������vw|������)5EGTURNB5)��������������������#)+5:==55)#:6BO[dht��th[OH?=?6:��������������������66BOORPOLEB?61126666��������������������)6BO_fg][WOK@��x}�������������zuuvx������������������������������������������������������������st�������������trqssY^gt�����������tg]YY#.05<EG<50-#`mz���������zmda[XY`��������������������������������������������

���������������������������):=51(�������������������������RUadlnnnaXUURRRRRRRR����

��������:BX_mz~����zmaTH?67:lmz|{zrmmkllllllllll#/9:3/)%#  fnsz{|��������{nlgff����������������������������������������`anquxz|�|znnia`^]``#%)/<HUacc^ZTQH</(##���������������������������������������������
#'&#
������')5BHKMLMB<5)#�����������9<GHKKMOH<9313999999Sahz���������zrmhaVS��)-36;6*)����)1679861+)	��������������|||��������������������tv������������tomnnt-/<HTU\__UUHE<5/,+--������������������

���������������������������	
!',/9<HLNLH</#
	�f�f�^�f�s�����������s�f�f�f�f�f�f�f�f�������������������������������������������������������������������������������ҿ����������������������������������������z�y�n�e�a�Y�a�n�t�zÂÇÈÇ�z�z�z�z�z�z�������d�O�A�!��8�Z������������������׽н�������s�t�����Ľݾ�!����������пy�`�C�8�=�G�`�y���ѿ��޿����������y��������������������������������������������������	������	������������#�)�)�6�B�O�R�O�O�E�B�6�)����àÓÇ�z�n�g�f�i�k�n�zÇÓÝäçíòìà���������������ûŻǻĻû����������������������ܺܺ������������������������������������߾۾����4�;�2�5�G�.��ʾ���������������*�>�C�F�D�C�6�*������������(�+�(��������������t�r�q�r�r���������������������Թ��������������ùϺ��'�1�3�������������������������)�0�/�(���������ŠŔōŅŇŎŔşŠŭųŹźŹ������ŹŭŠ�Z�X�T�Z�Z�g�s�y���������s�g�Z�Z�Z�Z�Z�Z�T�H�8�2�.�1�;�H�a�m�z�z���������w�m�a�T�;ʾ��������������׾���������������5�+�)����"�)�5�B�N�N�W�[�b�[�N�B�5�5�.�"������"�T�y�������������m�`�T�<�.�i�a�V�U�U�_�g�n�t�z�~ÇËØáàÓÇ�z�i���������������¾ʾ׾߾׾Ҿʾ���������������� ����)�*�+�*�%���������s�l�h�k�q���������������������������sƁ�u�h�e�\�b�h�uƁƎƚƳ������ƷƧƚƎƁ�������������������$�0�2�4�0�+������̺e�b�\�`�e�i�r�~�������������������~�r�e�/�(�(�-�.�/�/�;�<�H�T�a�e�k�k�c�U�H�;�/����¿¿»¿�����������������������������C�6�+�$� �*�-�6�O�Y�\�h�uƁƁ�u�p�\�O�C���ֻܻܻ޻�����������������������������
��#�0�9�<�=�6�0�,�#��
���<�;�9�<�H�I�U�^�b�n�{ņŅ�{�n�b�Y�U�I�<�I�I�F�I�V�b�i�o�z�o�b�V�I�I�I�I�I�I�I�I���������������ɺֺ����ֺҺɺ��������ʼ����f�[�j�����ʼӼ���'�/�*���ּ��B�:�@�I�W�l�|čĚĤĳĿ������ĹĦĚ�[�B���������ĿĿѿݿ߿�ݿտѿĿ��������������������ĿοѿѿۿѿĿÿ����������������-�-�!�����!�-�:�F�I�N�I�F�:�-�-�-�-���������n�����������	��!�%�����������5�3�-�5�A�J�N�R�N�A�5�5�5�5�5�5�5�5�5�5�U�H�L�U�a�n�v�zÇÉÇ�z�n�a�U�U�U�U�U�U�'��������'�,�4�@�B�J�K�C�@�4�'�'ĚĕčĖĚĦĳĻļĳıĦĚĚĚĚĚĚĚĚ���������������	������������������ʾɾ��ʾ׾�����	��	�	������׾ʾ�E�E�E�E�E�EyEwE�E�E�E�E�E�E�E�E�E�E�E�E�ŭŭŠŔŉœŔŠŭųŹ������żŹŭŭŭŭ�����������������������������������������m�k�a�^�`�h�j�m�z�������������������z�m�z�v�n�m�j�g�n�zÇÓàììíæà×ÓÇ�z����� ���������������FFFFF$F1F=FJFSFMFJF=F1F$FFFFFF�������������ùܺ'�L�Y�c�V�=�'���׹ù��'�$�����'�4�M�Z�f�l�o�k�f�b�Y�@�4�'�����u�s������������������¼�����������ĳĮĴĴĻ����������
���
���������ĳ�#���"�#�0�<�I�O�U�Z�b�f�b�U�I�<�0�#�#·²£¦²¿����������� ���������·�M�J�C�F�M�N�Z�f�s������y�s�m�f�Z�M�M�`�`�S�L�S�`�l�r�y���������y�l�`�`�`�`�`�g�f�g�s�w�������������������������s�g�g�����x�p�x������������������������������EED�D�D�D�D�D�D�D�D�D�D�D�E EEEEE C z p A } ] O @ D ( I p I A O M Q I A h ' 1 # W @ e P 6 ' l X g h C D 3 1 5 d Y = ~ N 4 P / Q W n $ & B l M ^ W V ! = T a  - : R n * u s u m    s  Z  �  �  n  �  �  A  u  �  �  {  �  �  �    !  �  �  K  ^  �  i  �    m  �  �  �  �  �  �  h  �     �  8  �  X  p  �  �  �  �  Y  �  �    �    �      J  �  0  �  n  =  �  �  )  g    2  �  ;  �      ?<�h%   ��o�o��o��+�1'��hs�C���9X��������9X���ͽ�{�49X��1���\�49X��P�49X�q���8Q�#�
�]/�,1���#�
��-�P�`�u�H�9�T���49X����P�`��o�]/�@��m�h�� žn���%�e`B�aG���h�Y��q���u�u��%��%���`�����\)���
��1��+���w�+��
=��j����P��9X�\�� Žě���E��+B%e�B��BxlB�~B!W%B&��B&�EB*�|BE�Bd�B#$B��B�B>�B��BV�B�B#�B(�B�AB�4B4'BսB!�B�Bt�B��B/B[�B�IB �dBA�B!nMBZ�B2�B	��B%��A��B�LBdeB#��B-C�Bb�B|GB(
B#�A��GA��B	B(�B�/B8�B�_B�BY=B\{B?�B:�B
�B��B-ABj�B�bBB �B
i:B�BT�B��B%�BB%F.BAVB�0B	8&B!��B'��B'?�B*ůB?kB�CB��BB�B�
B��B	PB:`B-�B5B?�B�BB��B@B�~B!iB� B��Br�B<B�oB�lB �-B�B ˧B/=B�B	�(B%B�A��B>�B�B#�#B-��B?�B�B?�B#��A��nA�z�B*B(�B<�B�B�*B��B;�B�B�B.�B
�DB�B�B@B)�B
��BB�B
@�B�IB>2B��BB�B��ADEA��$A�KAq�A��iA���A&3�Aq�kA�@�A�Aי A�H�@��@M��AX/�A���A�=@�]x?%:�A��&A��A�0�A�bAQ`'A�(�Aj��AȊOAN��A��
A�VB�0Bt�@�A�j�A�kQB�@�QA�5$A�M@Bi@5[�A�<Aݱ�Ax}�Ay+@v�XA�Y�A�g<A�v�@��A�r�BAU��C�2{A�@.A���A���Aɾ'A���C��V?]F@���@�"A�M�A줨A���A@BhAX�A�3y@�PC�6�AD��A��A�eAqA�W[A��A&��Ao�A�x�A��jA�i�AȑY@��@K��AY.A��{A�^�@�	�?�A��A���A���A�qAP��A��Al��A�meAN�gA�DA�<�B�B	6�?�Z�A���A��B>�@��BA�{iA�<BV<@3ۢA*A�t�Aw�Ay=�@{xA��A��yAœ�@�iA���B�AAV�C�:�A���A�lA��3A��A�^�C��?�_@���@��A�ӎA�mzA��A?@A4�A��@��C�                  <   }   ?      
            	   D         	   H                  
               4                                    ,   c            C      
   
   	   
      -      	               E   *         
                  )                  A   =   ;                     E            +                     )            )                                    ;   '            -                                          3                                                A      '                     #            !                                                                     !               #                                          #                              No}=N�ΧNf2�N�:�N(�UP�=�O�W�PO`N�9�N��:N�%�Ot�IN�Nbw�O�F�O��M�r�Nr��O�E�Ov�6O��N��pO�HhO;�oN��OBO%��N�FNL��O+1�OT�dOx fN���O^��N��hO'�&O��O$�uOYNI�N�N�O��~O�>6N�\�N#ߒN�>O��M�#�N84AN�%N�`NވN�? O�o�N��N�m�Oa��O-ХNlN��BO��'OE�uOINO,�N���OH=;OD�Nu��N�K-M��ObR  �  Q  �  3  w  u  
  �    �  �    #  �  o  \  �    	H  �  k  b      {  X  :  z  0  �  J    �  z  �  �  �  �  n  �  �  �  a  �  �  W  	�  �  �  L  �  >  Y  	�  �  c  �  �    (  �  �  �    \  v  �  �  �    7=+;�`B;�o;o:�o��`B���w����o�T���e`B�T����o��C��T���������㼴9X�49X���ͼ�`B���������'o�o�\)�ixս�����P����w�49X�#�
�49X�0 Ž0 Ž0 Ž]/���P�H�9�H�9�H�9��o�L�ͽT���T���T���]/�e`B�m�h�u�}�}󶽁%��%��o���罕���C���O߽��������������{�� žJ##022200#########!#'/<HUacaWUH</)'#!!knz��������zwnkkkkkkLN[gtxtqg\[ZPNLLLLLL���������������������
#0<{������U<#����#0<IUbgklibUI40*# ��������������������������������{{������>BNN[^ghgf[ZNEB:>>>>#/;<BA<<6/#"������)2.!������JO[_c]hjh[ONFHJJJJJJ�
!"��������&)/6BN[�������gNB5)&����� ��������������������������������w����������{wwwwwwww0BO[horruwvxth[QJ6/0v����������������}}v.5?BDNP[YNKB<52,)*..}�����������zy}}}}}})5BDFSTMB4)��������������������#)+5:==55)#NOO[ahiptzvth[SONJIN��������������������66BOORPOLEB?61126666��������������������)6BCMOTVROB62)#yz~������������zvvvy������������������������������������������������������������st�������������trqssbgt����������tgf`\\b#.05<EG<50-#`mtz��������zmea\YZ`��������������������������������������������

���������������������������&) ���������������������������RUadlnnnaXUURRRRRRRR����

��������;=BHamz����zmaTME?;lmz|{zrmmkllllllllll#/771/#  hnx{��������{nmihhhh����������������������������������������`anquxz|�|znnia`^]``#%)/<HUacc^ZTQH</(##���������������������������������������������
#'&#
������')5BHKMLMB<5)#�����������9<GHKKMOH<9313999999z������������{wsronz���")-122) ���),4666/)#���������������~~��������������������tv������������tomnnt-/<HTU\__UUHE<5/,+--������������������

���������������������������	
!',/9<HLNLH</#
	�f�f�^�f�s�����������s�f�f�f�f�f�f�f�f�������������������������������������������������������������������������������ҿ����������������������������������������z�y�n�e�a�Y�a�n�t�zÂÇÈÇ�z�z�z�z�z�z�������d�O�A�!��8�Z������������������׽������������������Ľݽ�����нȽĽ������y�^�R�K�I�O�`�m�������Ŀѿ޿�޿ѿ����y�����������������������������������������������������	������	���������)��#�)�*�6�B�O�P�O�M�C�B�6�)�)�)�)�)�)àÓÇ�z�n�g�f�i�k�n�zÇÓÝäçíòìà���������������ûŻǻĻû����������������������ܺܺ����������������������������׾ʾǾʾؾ����	�� ���������������������*�3�6�=�6�/�*�������������(�+�(��������������t�r�q�r�r���������������������˹������ùϹܹ�����$�)�)���������������������������)�0�/�(���������ŔŎŇņŇŐŔŠŭŸŹ��������ŹŭŠŔŔ�g�[�Z�W�Z�]�g�s�v���������s�g�g�g�g�g�g�T�H�9�3�/�3�9�;�H�a�m�z��������v�m�a�T�;ʾ��������������׾���������������5�+�)����"�)�5�B�N�N�W�[�b�[�N�B�5�5�T�K�G�@�G�P�T�`�i�y�����������|�y�m�`�T�i�a�V�U�U�_�g�n�t�z�~ÇËØáàÓÇ�z�i���������������¾ʾ׾߾׾Ҿʾ��������������������&�&� ������������z�v�u�������������������������������Ɓ��u�h�f�^�d�h�uƁƎƚƳƿƿƴƧƚƎƁ���������������$�0�2�4�0�*����������̺e�b�\�`�e�i�r�~�������������������~�r�e�/�(�(�-�.�/�/�;�<�H�T�a�e�k�k�c�U�H�;�/����¿¿»¿�����������������������������6�0�*�&�*�5�6�C�M�P�\�h�p�p�k�h�\�O�C�6���ֻܻܻ޻�������������������������������
���#�0�;�5�0�+�#��
���<�;�9�<�H�I�U�^�b�n�{ņŅ�{�n�b�Y�U�I�<�I�I�F�I�V�b�i�o�z�o�b�V�I�I�I�I�I�I�I�I���������������ɺֺ����ֺҺɺ��������ʼ������ʼټ������$�$�������ּ��[�U�O�M�T�c�tčĦĳĶĺĶĮĦĚčā�h�[���������ĿĿѿݿ߿�ݿտѿĿ��������������������ĿοѿѿۿѿĿÿ����������������-�-�!�����!�-�:�F�I�N�I�F�:�-�-�-�-�����������������������	�����	�������5�3�-�5�A�J�N�R�N�A�5�5�5�5�5�5�5�5�5�5�U�L�N�U�a�n�q�x�n�a�U�U�U�U�U�U�U�U�U�U�'�!�����'�0�4�;�@�H�I�A�@�4�'�'�'�'ĚĕčĖĚĦĳĻļĳıĦĚĚĚĚĚĚĚĚ���������������	������������������ʾɾ��ʾ׾�����	��	�	������׾ʾ�E�E�E�E�E�EyEwE�E�E�E�E�E�E�E�E�E�E�E�E�ŭŭŠŔŉœŔŠŭųŹ������żŹŭŭŭŭ�����������������������������������������m�k�a�^�`�h�j�m�z�������������������z�m�z�v�n�m�j�g�n�zÇÓàììíæà×ÓÇ�z����� ���������������FFFFF$F1F=FJFSFMFJF=F1F$FFFFFF�������Ϲܹ���3�E�L�N�L�C�3�'���ܹϹ��4�0�'�!�����'�4�@�M�W�_�a�]�Y�M�@�4������{�{���������������������������������ļĿ���������������
���������������#���"�#�0�<�I�O�U�Z�b�f�b�U�I�<�0�#�#·²£¦²¿����������� ���������·�M�J�C�F�M�N�Z�f�s������y�s�m�f�Z�M�M�`�`�S�L�S�`�l�r�y���������y�l�`�`�`�`�`�g�f�g�s�w�������������������������s�g�g�����x�p�x������������������������������EED�D�D�D�D�D�D�D�D�D�D�D�E EEEEE C z p A } ] 4 6 C ( B p I A ; D Q I = h + 3 " W @ B P 6 / A W c h C D ) 1 0 d Y = \ H 4 P / / W J % & B l M ^ W V ! = T d  " 6 R n * u s u m    s  Z  �  �  n  �  9  j  �  �  �  {  �  �    `  !  �  n  K  +  �  K  �    I  �  �  b  u  �  H  h  �     f  8  a  X  p  �  �  �  �  Y  �  �    I  �  �      J  �  0  �  n  =  �  *  �    o  2  �  ;  �      ?  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  DX  �  �  �  �  �  �  �  �  �  �  �  �  p  W  >  -         �  Q  G  =  1      �  �  �  �  �  �  �  �  a  4    �  �  q  �  �  �  �  �  �  �  �  �  u  `  J  7  &      �  �  z  J  3  *           �  �  �  �  �  �  �  �  �  �  �  �  p  \  w  u  r  o  i  Y  I  8  (      �  �  �  �  �  �  }  r  g  u  [  !  �  �  j  I     �  �  |  6  �  �  c    �  b  �   �  @  �  	g  	�  	�  	�  	�  	�  	|  	w  	�  	�  	�  	T  �    8  �  1    �  �  �  �  �  �  �  �  �  �  �  {  I    �  �  ;  �  G    �  �  �  �  �    �  �  �  �  �  �  �  �  ;  �  G  �    y  �  �  �  �  �  �  �  �  �  z  h  U  B  -    �  �  �  �  �  �  �  �  �  }  h  S  =  '      �  �  �  �  �  C    �  �      �  �  �  �  �  �  �  n  <    �  �  `  +  �  �  �  �  #      �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  f  Z  N  B  6  +  !  �  y    �  �  �  �  �  �  �    I  e  m  X  6    �  g  �  �  �  f    (  C  S  Z  \  [  U  K  ;    �  �  ~  @  �  �  �  �  j  �  �  �  �  �  �  �  �  �  �  �  �  �  t  b  P  >  ,        z  t  j  \  N  C  7  /  '  $  '  +  <  M  D  3      �  �  f  �  	  	0  	D  	C  	   �  �  I    �  z    �  �  �  �  p  �  i  <  $  �  �  �  �  ;  �  j  �  �  �  �  �  c  =    �  b  g  j  ]  P  @  /    	  �  �  �  �  �  �  u  Z  <    �  _  `  b  `  Z  Q  B  /    �  �  �  �  ^  -  �  �  �  k  s        �  �  �  �  �  �  q  B    �  �  T  �  �  o  �  �        �  �  �  �  �  �  j  <    �  �  M    �  �  :   �  {  t  m  g  `  ]  Z  R  G  9  '    �  �  �  �  r  @  �  �  F  9  "  )  B  F  D  @  P  L  6    �  �  �  U    �  l   �  :  "  
  �  �  �  �  �  n  Y  E  3      �  �  �  �  l  4  z  v  q  l  `  S  F  6  %    �  �  �  �  �  P  !   �   �   �  #         %  -  (      �  �  �  �  �  �  �  �  �  �    W  �    A  g  �  �  �  �  �  �  s  ?    �  k  	  B  Z    H  I  D  9  ,      �  �  �  �  p  J  "  �  �  �  w  M  '        �  �  �  �  P    �  �  Z    �  p    �  0  �    �  �  �  �  i  O  1    �  �  �  �  �  g  Q  @  3  3  =  M  z  r  h  \  M  <  ,      �  �  �  �  �  g  8    �  �  �  �  �  �  �  �  �  �  �  �  w  l  b  X  L  A  5  (         �  �  �  �  �  �  �  �  �  �  �  k  H    �  �  .  �  6  �  �  �  �  �  �  �  n  Y  H  7  '    �  �  �  �  W  (   �   �  �  �  �  �  �  �  �  x  Z  4  
  �  �  ?  �  l  �  _  �  6  n  d  X  C  ,    �  �  �  �  �  �  �  m  P    �  �  s  ;  �  �  �  �  �  �  ~  s  h  ^  P  ?  .      �  �  �  �  �  �  �  �  �  �  �  �  �  x  `  F  '    �  �  �  a  #  �  �  T  j  6  �  �  �  �  �  �  �  v  L    �  v    �    D    �  b  �    H  \  _  T  5  �  �    p  �  
�  	�  �  U  �  �  �  �  �  �  �  q  T  5    �  �  �  �  c  7    �  j  �  �  �  �  �  �  �  �  �  Y    �  �  �  �  �  p  U  :    �  �  W  O  F  =  1  %         �  �  �  �  �  �  �  �  �  �  �  �  	A  	k  	�  	�  	�  	p  	I  	  �  �  T  �  �  	  �  �  6  l    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    *  R  =  D  K  J  G  >  3  (        �  �  �  �  �  �  �  `  <  �  �  n  [  I  7  %    �  �  �  �  �  f  B  
  �  �  ]  *  >  <  9  *      �  �  �  �  �  �  �  x  e  P  8    �  �  Y  F  3  "      �  �  �  �  �  �  �  b  2    �  �  �  �  	�  	�  	a  	F  �  �  �  i  )    �  �  �  W    �  2  �  �  )  �  �  �  �  �  �  �  �  l  R  8      �  �  �  �  r  T  6  c  b  `  Y  O  D  3  "    �  �  �  �  �  �  |  p  X  -    �  �  �  l  B    �  �  }  X  :    �  �  y  8  �  �  i  (  �  s  W  <     �  �  �  {  J    �  �  j  %  �  �  ,  �  x    
      �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  m  (  	  �  �  �  �  e  F    �  �  e     �  �  A  �  �  6  �  /  �  �  �  �  �  �  �  �  �  �  �  w  -  �  	  e  �  �  �  d  |  �  �  �  �  �  o  S  0  
  �  �  e    ~  �  [    �  �  �  �  �  �  �  �  �  �  �  �  �  n  G    �  �  �  E  	     	              �  �  �  �  M    �  �  V  %    �  \  R  G  >  6  ,  !    �  �  �  �  s  N  &  �  �  �  v  J  v  j  R  6    �  �  �  �  �  m  7  �  �  q    �  L   �   i  �  �  �  �  m  M  +    �  �  m  (  �  �  9  �  u  <    �  �  �  �  �  �  �  �  �  m  P  1    �  �  �  �  �  �  x  U  �  �  �  �  �  ~  q  d  R  =  !  �  �  |  #  �  �  N    �        �  �  �  �  �  �  �  u  Y  =  $    �  �  �  �  �  7    
�  
�  
9  	�  	�  	F  �  �  *  �  %    �  7  �  c  �  �