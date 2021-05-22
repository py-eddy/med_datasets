CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���S���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       Pê      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���T   max       <�h      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F'�z�H     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @v�          	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�(`          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��-   max       <ě�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��O   max       B0+y      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0I�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?^��   max       C���      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?K��   max       C���      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          l      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P?\�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�m��8�Z   max       ?��҈�p;      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       <�h      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F'�z�H     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @v~fffff     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @O�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�&�          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�   max       ?�ۋ�q�     �  T            a         
      
   <            &   l      '   ]                  :                  -   #       +   L            !   3            '      ,      
   &      c                #            	   N�9�Oُ�N��6PêN�?NI�DN���O��Ny�8O��N�ĮN���NU�P8PD"�OZ1O��GPN<�NMY�N
c�O*�O*_NK�O��N��OML�N0��N��]O*�qP;��O�k�P(�O~�Pb�O�sqN���O�U1O#�8P �|O&�pN�kaN���O��;O��PJRiNn��N�P,Ow�OK��O��9O�nxN!�O��9N���O�_�N���M���O�N���O�C<�h<u<e`B<D��<t�<o%   ��o�D����`B�o�49X�49X�D���T���u�u��o��C����㼣�
��1��9X��j�ě��ě��ě����ͼ��ͼ�`B�o�C��C��\)�t����#�
�'',1�0 Ž8Q�<j�@��H�9�H�9�H�9�H�9�Y��aG��e`B�e`B�e`B�m�h�m�h�u��������hs���T #*/8986/#�)0:AEB5)�����������������������������-:2<H<7�������HHOUYabglnpnaUSHHEHH>BHNO[^b[WOOIB>>>>>>��������������������
#0566420#

��
#/9/'%#
��������
#/6>@@=5/#��� %*-46CKOMHC=6*��������������������#08:30$#��������������h�������������}u~whh)5@BGKIB5))6O[n{��h[O6)��
#Ubp}�znUI@<0����������������������6BNOYQNLB@6666666666������� ���������w|���������������zyw���������������������������tk]WT_g������fgt���������ztogffff
#/=@@?<8/(#@BGOTWOB<9@@@@@@@@@@057BFNORUSNB95310000��������������������mt�������������zqfkm`gz�����������|mb\\`BOTalw�����zmaTH><<B��������������������#/?>Daz��znaH/#��������������������{���������������xw{{���������������������#)*054)����!1BNt��������gNB(!���
#'.#
�������)*/5BFMB@5)'��������������������nz�������������smhin!4IUbntsg`ZI<0"i}�������������tefbiaalnz�����znhaaaaaaa������������������������������������������������������������
#/<HTSSOH</#�������
��������
%'!









!)06O[htuwtrh[B:0( !�����utotw����������LQ[gt���������|tg[NLlnqv{���������{yonll����������������������������������������������������������S[]gt�������xtgd[YRS�������߾�����	��!����	���������A�+��!�,�5�Z�g���������������{�s�g�Z�A�
� �����������
����!���
�
�
�
�
�
�(����-�N�g�����������������������N�(�5�3�(�"�(�3�5�A�N�U�Z�^�_�[�Z�N�N�A�5�5�����������������������������������������f�e�[�Y�P�M�C�M�Y�f�h�q�r�}�w�r�f�f�f�f�Z�S�M�L�I�G�M�Z�f�s�����������s�j�f�Z������������������������������������������ŹŲŰűŴŹ�����������������������ƿ��	��������ھ����	���� ����H�D�<�6�<�@�H�U�_�a�f�n�o�n�m�c�a�U�H�H�������!�-�!�!������������"��������������	�H�T�[�Z�`�b�a�T�H�0�"����5�@�L�r���������������~�r�Y�3���B�?�7�9�B�E�N�[�g�j�j�l�h�g�[�N�B�B�B�B�-�,�:�:�M�]�l�x����������������x�S�F�-���y�x�k�l�y���Ľݾ��+�-����Ƚ�������������������������������������������꿒��������������������������������������àØÓÓÑÍÍÓàãìôù������ýùìà�������������������������������������������������������������������������������������� ��������ùöóðõù���ҽ������������	��������������������$�)�6�B�O�[�d�h�m�h�_�b�[�O�B�6������������ĿϿ̿Ŀ������������������������������������������������������������Ż����v�j�e�e�l�x�����������ǻû���������ĲĿ��������Ĺľ������0�B�@�G�=�0��ĿĲ�����ܿտֿݿ�����(�;�I�N�X�N�5�������������������	��"�4�>�<�4�%�������Ç�|�z�w�yÇÓàìù������������ìàÓÇE�E�E�E�E�E�FFF1F=FHFPFNF>F7F.FE�E�E��������������*�6�B�\�h�t�u�k�O�6�*�����¿º¿��������������������������������������ƳƧƕƧƺ������� �����������h�b�\�`�h�q�tāčĚĠĞĚĖčĉā�t�h�h���پ־߾���	��.�:�D�F�C�3�(��	����!����!�%�-�-�:�@�S�T�`�Z�S�M�F�:�-�!�
���
���!�#�/�;�<�>�=�<�3�/�#��
�
�s�q�f�d�[�f�s�|���������������s�s�s�s�عʹƹȹչ����'�@�L�W�L�E�5��������������t�i�b�`�g�s�����������������������g�B�=�B�W�t²��������������¿�z�y�m�l�j�m�p�z�{���������z�z�z�z�z�z�z�H�B�A�C�H�O�T�a�c�h�k�i�a�T�H�H�H�H�H�H�j�_�M�_�j�z�������������»����������x�j�ƺ����ɺκֺ��������������ֺ�E*E EEEE	EEE*ECEPE]EmEoEjEaEPECE7E*�����������������������������������������A�6�;�A�N�Z�b�Z�O�N�A�A�A�A�A�A�A�A�A�A�g�a�g�i�{��������������������������s�g�a�b�W�H�E�<�7�/�%�#��#�&�/�1�<�H�U�a�a��ŭūūťŘŖŢź���������������������Ƽ'� �������'�4�@�M�M�M�B�@�4�1�'�'�����������������������������������������ּ˼ʼļʼּؼ��������
��������I�G�B�=�4�0�*�0�5�=�I�J�V�W�a�b�e�b�V�IĿĴĳİĴĺĽĿ����������������������Ŀ 0 ) / = G Z v + u > Y 1 4 0 D * q \ ) r N G @ 2 ` N : @ p A M E d h \ : J C I 5 P K V j Q ; > ? .  p L h g F G  v \ <  �  �  �  �  �  v  �  <  �  Y    �  g  �  �  8  D    b  V  �  �  1  �  �  �  H  �  �  @  R  �  :    �    �  c    p    }    H  �  �      �  �  �  L  k  L  1  �  q  d    g<ě��#�
%   ��{;o;��
�t���o�D����+�u���ͼe`B�L�ͽ����Y���;d��`B��9X��w�'������T��`B�0 ż�h���@���t���+������-��G���%�0 Žq�����P��j�Y��H�9�P�`��1�y�#��j�Y��m�h�� Ž����-��9X�u���㽛�㽾vɽ�7L��C����������ȴ9B��B�?B~9B�B _B��B � B%!_B0>BdJB0+yB!iB%��B$B ��B�qB[�B&�B��B��BQ�B;dB�B
l�B
B�B��BwCBƆB!tBq^B W�A��OBWfB�B�ZBn$B��BgB	MiB$+�B�.B �=B�B&s�B
��B�B�&BAB��BғB�B��B�oB
��B�B)nB+~�B,�>B��B	�hB5B�BBD�B��B�\B=�B ��B%?B<B��B0I�B!?EB%��B��B ��B�LB�5B'?�B�HB>3B��B<�B;�B
BNB
�B�
BG�B��B ��Bv0B K�A���B�B:0B�DBUBKmB�qB	B�B#�rB�B ŉB�B&zfB]B�~B�B>�B��B��B>_B�cB�DB
�B?B)�B+��B-�B0�B	@�AY�{A�z�A�wA��1A���A���@�9jAA�FA�bFA���AY��AŐ�A
��A�X?�$�A�	D@�0zA%��A�bAru�A��A�A)A��MA�RA/��A�EAw�.A���@�F�A�w@A���A���A�g�C���B  qA���B��A�I�A[��@{;�A���AC�[?^��A��A�4A��~A��j@��@F��C���A���A�1A���A���A���@��@�ADEB;�A�p�AY!A�}9A��YA�DA�>A�~@��AA�A�u�A��.AY~A���A
�A���?׈�A��X@��A$��A�{@AsOĀ�A��A���AЌA1NA��Aw4�A���@��SA�QA��OA�?A�)C���A�ɆA�tqB^A܀yA[
k@vnA�%�AE�.?K��A���A�c�A���A��\@��"@Dd_C��A���A���A��A���A�u�@�f,@�jA�sB3�A��y            b         
      
   <            '   l      '   ^                  ;                  -   #       ,   L            "   4            '      ,      
   &      d   !            $            	         !      ?                              )   3      %   7                                    /   #   '      +   !      #      )            %   '   3                  %      !      #                           -                              #            !                                    -   #   '      '   !      #                  !      1                  %                           N�9�OP�IN��6P?\�N�?NI�DN�7kO��N6�POa�_N�ĮN���NU�P	XOT��OZ1O{�HO��NMY�N
c�O�bN��NK�O�NfN��OML�N0��N��]OV�P&�mO�k�P(�O_)jOݳ�O�sqN���O�U1O�|O�dO&�pN�kaN���O��
OZ�P=\�Nn��N�P,Oa��OQOi�O�nxN!�OgtN���OU�N���M���O�N���O�3  �  �  �  �    �  #  �  �  
y  _  !  �  �  
  E  $  	q  9    �  x  �  
G  �  �  �  �  D  �  (  �  	�  
�  �    I  �  �  g  l  <  +  s  X  t  Q  �  G  .  �  9  �  �  7  �    [  5  K<�h;��
<e`B��1<t�<o��o��o��o��o�o�49X�49X�u��C��u��j�aG���C����㼬1��`B��9X��/�ě��ě��ě����ͼ������o�C���P�49X�t����#�
�49X�}�,1�0 Ž8Q�H�9�L�ͽP�`�H�9�H�9�P�`�m�h��j�e`B�e`B�m�h�u��\)�u��������hs��� #*/8986/#��&)+-)���������������������������
"
���������HHOUYabglnpnaUSHHEHH>BHNO[^b[WOOIB>>>>>>��������������������
#0566420#

�
#/0/%#"
��������
#/2;<<9/#
���� %*-46CKOMHC=6*��������������������#08:30$#���������������������������������)5@BGKIB5)#)6BOWfrrohOB6)-2<IU\dgmnnlgbUI?6--��������������������6BNOYQNLB@6666666666�����������������������������~����������������������������Xdt������������tm_ZXfgt���������ztogffff
#/=@@?<8/(#@BGOTWOB<9@@@@@@@@@@057BFNORUSNB95310000��������������������mz�������������uhhm`gz�����������|mb\\`BOTalw�����zmaTH><<B��������������������#/88?arvqaH/#��������������������{���������������xw{{���������������������!)).30)���@N[gt��������tg[OD?@���
#'.#
�������)*/5BFMB@5)'��������������������knz������������unik #'0;FIUa``^\VI<0# kt������������tggckaalnz�����znhaaaaaaa������������������������������������������������������������	
#/:<??<<0/#
	�������
��������
%'!









(*6OW[htvtoh[B<2)%"(st������������xtssssy��������������xutylnqv{���������{yonll����������������������������������������������������������T[_got�������tge[YRT�������߾�����	��!����	���������Z�N�9�5�/�3�5�A�N�Z�s�x���������y�s�g�Z�
� �����������
����!���
�
�
�
�
�
�N�>�=�F�Q�g�������������������������s�N�5�3�(�"�(�3�5�A�N�U�Z�^�_�[�Z�N�N�A�5�5�����������������������������������������r�i�f�]�Y�Q�M�E�M�Y�f�g�p�r�{�u�r�r�r�r�Z�S�M�L�I�G�M�Z�f�s�����������s�j�f�Z����������������������������������������žŹŶųŴŷŹ�����������������������ƿ��	��������ھ����	���� ����H�D�<�6�<�@�H�U�_�a�f�n�o�n�m�c�a�U�H�H�������!�-�!�!������������/�"�������������	�/�H�T�Y�X�^�a�_�T�H�/�@�3�3�6�@�L�Y�e�r�y�~�����~�w�r�e�Y�L�@�B�?�7�9�B�E�N�[�g�j�j�l�h�g�[�N�B�B�B�B�S�F�C�E�J�R�_�l�x�����������~�y�z�x�l�S�����~�{�}���������нݽ��� ���н�����������������������������������������꿒��������������������������������������àÙÔÓÒÎÎÓàìòùÿ����üùìàà�����������������	������������������������������������������������������������ùöõòú�����������������������Ž������������	��������������������$�)�6�B�O�[�d�h�m�h�_�b�[�O�B�6������������ĿϿ̿Ŀ������������������������������������������������������������Ż����x�k�f�l�x�������������Żû���������������������Ļ��������0�<�<�C�<�0�����������ܿտֿݿ�����(�;�I�N�X�N�5�������������������	��"�4�>�<�4�%�������Ç�~�z�x�zÇÓàìù������������ôàÓÇE�E�E�E�E�E�E�FF$F1F=FFF9F3F+FE�E�E�E��������������*�6�B�\�h�t�u�k�O�6�*�����¿º¿��������������������������������������ƳƧƕƧƺ������� �����������h�d�^�b�h�t�tāčĚĞĝĚĔčăā�t�h�h�������������	��"�.�0�2�0�*�"��	���!����!�%�-�-�:�@�S�T�`�Z�S�M�F�:�-�!�
���
���!�#�/�;�<�>�=�<�3�/�#��
�
�s�q�f�d�[�f�s�|���������������s�s�s�s��ܹ͹ʹιٹ����'�3�@�G�B�3�'�������������������}�t�������������������������y�g�N�E�E�Y�t¯��������������¿�z�y�m�l�j�m�p�z�{���������z�z�z�z�z�z�z�H�B�A�C�H�O�T�a�c�h�k�i�a�T�H�H�H�H�H�H�x�n�l�b�l�m�|�������������������������x�ֺ˺ɺúƺɺֺ����������������E7E5E*E$E E"E*E-E7ECEPEPE\E]E_E\EYEPECE7�����������������������������������������A�6�;�A�N�Z�b�Z�O�N�A�A�A�A�A�A�A�A�A�A�g�g�k�|�����������������������������s�g�/�-�$�(�/�3�<�H�U�a�V�U�H�C�<�1�/�/�/�/ŭũŦŦŭŲŹ����������������������Źŭ�'� �������'�4�@�M�M�M�B�@�4�1�'�'�����������������������������������������ּ˼ʼļʼּؼ��������
��������I�G�B�=�4�0�*�0�5�=�I�J�V�W�a�b�e�b�V�IĿĵĳıĳĵĽĿ����������������������Ŀ 0 ) / > G Z q + } 1 Y 1 4 -  * m V ) r O 7 @ . ` N : @ o D M E ` d \ : J E 7 5 P K H N M ; > 3 ,  p L i d + G  v \ 7  �  �  �  Y  �  v  �  <  �  �    �  g  g  �  8  K  �  b  V  r  �  1  _  �  �  H  �  �  �  R  �    �  �    �  ?    p    }  �  �  }  �    �  1  O  �  L  (  �  �  �  q  d    @  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  �  �  �  �  �  �  �  �  �  z  m  ]  M  =  $  	   �   �    %  C  b  {  �  �  �  �  �  u  P  #  �  �  �  5  �  J   �  �  x  n  h  l  k  f  Y  <    �  �  �  k  ;    �  �  �  Z  �  E  ~  �  �  �  �  �  �  �  �  �  >  P  x  }    g  U  �        	    �  �  �  �  �  �  �  n  W  N  X  a  W  F  5  �  �  �  x  p  i  a  Y  Q  H  @  8  0  %       �   �   �   �       #        	    �  �  �  �  �  �  �  �  �  �  �  �  �    t  f  W  F  4  #    �  �  �  �  }  M    �  y     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  q  j  	�  
@  
k  
x  
o  
a  
M  
2  
	  	�  	�  	?  �  g  �    I  a  !  �  _  V  M  D  ;  2  (        �  �  �  �  �  �  �  {  _  C  !    
  �  �  �  �  �    U  &  �  �  �  ^  C  �  �  =  �  �  �  �  �  �  �  �  |  v  p  i  c  ]  V  O  G  @  8  1  )  �  �  �  �  �  p  P  .  	  �  �  �  @  �  �  [  &    p  �  �  	  �    x  �  	?  	�  	�  

  
  	�  	�  	p  	  �  �  �  �  �  E  ;  2  )  !        �  �  �  �  �  c  B    �  �  �    �  �      $  "    
  �  �  �  �  �  �  k    �  �  Y  �  [  �  t  �  		  	2  	S  	j  	q  	`  	6  �  �  :  �  �    p  p    9  B  N  c  n  q  p  m  h  b  [  P  B  -    �  �  �  q  E      �  �  �  �  �  �  �  �  �  �  ~  n  W  A  *     �   �  �  �  �  �  �  �  �  v  R  *  �  �  �  e  +  �  �  _    �    =  B  Q  e  q  w  x  w  s  l  c  P    �  9  �  E  �  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
4  
G  
F  
4  
  	�  	�  	�  	�  	?  �  y  �  ~  �  r  �  �  N  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    n  ^  M  �  �  �  s  P  -  =  4    
  �    �  �  �  �  �  ]  0  �  �  �  �  �  �    s  h  \  P  E  9  ,                �  �  �  �  �  �  �  �  x  ^  5    �  �  ^  '  �  �  �  U  A  B  +    �  �  �  �        �  �  s  3  �  �  �    (  <  �  {  e  Q  @  -      �  �  �  9  �  �  �  �  [  �  c  (    
  �  �  �  �  ^  ,  �  �  �  L    �  �  1  �  !    �  d  O  E  .      �  �  �  A  �  �  w  B  �  �  7  �    	�  	�  	�  	�  	�  	~  	]  	1  	  �  �  Y    �  0  �    d  N  �  
j  
�  
�  
�  
  
E  
5  
x  
i  
j  
I  
  	�  	5  �  �  3  s  R  H  �  �  �  �  �  �  �  _  ,  �  �  n  '  �  �  S  �  �  �      
    �  �  �  �  �  �  �  �  �  �  h  J  ,     �   �   �  I  3  %    !    �  �  �  �  c  7    �  �  �  [    �  �  �  �  �  �  �  }  e  H    �  �  n  *  �  �  &  �  $  N  �  ;  e  �  �  �  �  �  �  �  �  �  �  `  '  �  }  �  T  �    g  \  R  K  :  #    �  �  �  �  �  h  O  9  ,  4  U  w  �  l  c  Z  Q  H  ?  5  .  '  !                	    �  <  0  $      �  �  �  �  �  �  {  ]  D  4  $        �  �  �  (    �  �  �  �  u  L  !  �  �  {  !  �  G  �  7  �  s  ]  [  e  r  r  o  l  h  b  Z  P  <  #    �  �  �  r  G  :  X  S  D  /    �  �  �  �  o  C    �  �  A  �  %  @  5  t  o  j  e  `  X  J  <  .       �  �  �  �  �  �  �  �  �  Q  G  =  2  (        �  �  �  �  �  m  H  "  �  �  �  q  -  �  �  �  �  n  G    �  �  e    �  n    �  .  �  �  �    '  8  B  F  B  4  "    �  �  �  �  G  �  �    �  �  f  �  �  �  ^  �  �    ,    �  �  r    �    Q  V  
  N  �  �  �  �  �  �  r  M  *  	    �  �  �  �  O  �  �  a  �  C  9  '      �  �  �  �  �  �  n  Y  C  .    �  �  �  �  �  �  �  �  �  �  �  x  Y  5  	  �  �  k  3  �  �  �  .  �    �  �  �  �  �  �  u  d  S  M  >    �  �  �  `  5    �  t  �  �  �  �    #  5  3  (    �  �  �  x  ?    �  ;  �    �  �  �  �  �  |  r  h  ^  S  G  <  0  $        �  �  �          $  *  /  0  .  ,  *  (  %        �  �  �  �  [  K  ;  -  $  +  -  !    	    6  :  )    �  �  |  H    5  !    �  �  �  �  �  i  K  /    �  �  �  �  �  �  �  �  0  H  >  5  >  ?  *  
  �  �  �  �  �  v  Q  (     �  b  