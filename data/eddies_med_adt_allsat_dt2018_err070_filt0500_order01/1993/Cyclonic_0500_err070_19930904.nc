CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��vȴ9X     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��+   max       P*8     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���w   max       =+     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F]p��
>     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @v{
=p��     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�w        max       @��          0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       <�9X     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B1�n     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B2�     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�1~   max       C���     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@�b   max       C��-     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          >     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��+   max       P*8     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?��u&     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       =+     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F]p��
>     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vy\(�     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�w        max       @��         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >P   max         >P     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?s@N���U   max       ?��u&     �  b�                                 	                           
                                       
            	   
                      
      5      
            %            +   >      
      !                     ,                      N��N��N���N$,�N1&lNrwhN�i&Nh��P=*N��RO;��N��P��N��N�˗N��NළO?�N�N�j�O��M��+NO,�P*8N�,�P��N�0wNm��O�9O���N�� O!`N��CO#�N���OV�PN���N(�OF�6O�SiO�4"O#�NV�}O9~"O;�O43+Oln�O?�.N�C2N�/�NQ�OiZ�OJ�O{�N j�O�0O�^6O.N>N�iN4�kO�"N�'�O��N�k�N��oN���OBVO�R�N���OS��O)ՇM�\�O��-O+8/O5I�=+<�t�<D��<t�<t�;�`B;�`B;ě�;D���o��o��`B��`B�D���e`B�u��o��o��C���C���C���t���t���1��1��1��1��9X���ͼ�`B��`B��h�������+�+�t���P��������w�#�
�''''''0 Ž0 Ž8Q�@��H�9�L�ͽP�`�P�`�Y��]/�q���u�}󶽁%��+��+��hs��hs���������������㽝�-���-���w���w����������~{��������//<HSQH<7///////////:<HUUXZUH<<:::::::::%)6=BCB@<6)%%%%%%%%%OOY\houyuh`\VOOOOOOO��������������������3<IUbkcbZUMI<:853333����������������������5BN[grtrg[)���������������������������������������������������������������HKLUaz��������zTD>>H')6BFOPTOHB6)�����������������
***
���fgt�������~tgc]cffff
#&/4<HPRHF</#"
���������������������������LX[ghtw����th[RMOJL����������������������������������������/7;?HPRPNMKHC;/,)((/ETaz����������zaTGAE#)35@BJHB?>5/)$"####BDNg�����������tgT<B������������������������������������������


�������������������#/<CHUKH</$#������������������}�chhmt�������thcccccc #&./<>@AC?;1/# Z[gt�������tge[YZZZZ��������������������
#/28/#
	
#&#!
								loswz�����������~zml������������������������
 
���������������������������������������������������������������������������������������
#'159:2/#
AHNUansxyzzsnaUOH@?A��
#0<IPQNI<90#
�dno{�����{tneddddddd��������������������#%"��������������� ����wz�����������������wltz~�����tsmllllllll&0<N\aebVRIF<0#"$08FIUbioolbU<0)$""����������DHPU[`a_VUHE>@DDDDDDnt~�����xtllnnnnnnnn��������#+(�������������������������JOW[[\ghttxzttg[NMKJV[[dhikptutmh[RPVVVV�����������������vz�����������zzzusvv�������������������������������������������������������������������
����������������������������()57AA54)(((((((((((I[gt���������t][MIFIggmst�����������tjgg���������������������ּμּټ�����������ּּּּּּּ��$�$��!�$�0�5�3�0�$�$�$�$�$�$�$�$�$�$�$�;�8�1�:�;�H�T�`�`�U�T�H�;�;�;�;�;�;�;�;�/�(�#�#�!�#�$�/�<�E�@�<�/�/�/�/�/�/�/�/�����������������������������������������H�E�H�N�U�`�a�c�n�p�t�n�a�U�H�H�H�H�H�H�ʼ������żʼּ׼�������ּʼʼʼ�¦¬¬§¦�Ŀ��ƿ��������������ѿݿ����������ѿ��T�O�I�J�T�a�m�r�t�m�a�[�T�T�T�T�T�T�T�T�;�9�3�4�4�6�;�>�G�T�`�n�u�o�d�`�\�T�G�;�������������������������������������������������~�������������������������������m�l�`�\�U�\�`�j�m�y���y�u�w�n�m�m�m�m�m������|�~������������������������������T�Q�G�E�;�3�.�*�'�.�;�<�D�G�N�T�Z�[�U�T��������������������	���	������������¿µ²«²¹¿������������������������¿ŔŒŋōŒŔşŠŭŹ������ŹŵŭūŠśŔ������������������������������������������������������������Ľݽ���ݽн����������������������������������������������M�E�M�Z�d�f�s�v���������s�f�Z�M�M�M�M���������q�s����������������������������������������(�7�?�I�W�Z�U�5������������������������������������������������������������������)�B�A�:�,����չ������������ùϹֹ���ܹϹƹù��������F�<�:�8�:�F�S�_�f�l�o�l�_�S�F�F�F�F�F�F��	������������������������	�����àÓÎÇ�z�a�a�p�yÇÓÝãìñùûøìà�H�D�A�A�H�M�U�a�b�c�h�a�a�U�H�H�H�H�H�H���������������������������������������¿³³¿���������������������������N�F�B�5�4�1�5�B�N�[�g�t�t�g�[�N�������������	���!�����	�����������	������
��.�;�A�C�D�;�2�.�"������� �(�)�6�>�=�6�1�)�!��������ÿùðóù�����������������������������H�;�"���������	��!�$�/�;�F�H�T�X�T�H�����x�e�Y�[�f�l�x��������������������������������!�-�:�D�O�V�S�F�:�!�����e�]�h�e�Y�V�L�E�B�L�Y�e�r�x�y�v�t�u�r�eŹŹűűŹ������������źŹŹŹŹŹŹŹŹ�������������� �*�6�=�C�D�C�6�+��ŭŪŭŭŸŹ����������������������ŹŭŭE�E�E�E�E�E�FFF$F1F1F)F$FFFFE�E�E湪���������������ùϹܹ�����޹ù��������������������������������������������g�[�Z�P�R�Z�g�r�s�|�������s�g�g�g�g�g�g�H�L�N�M�H�;�/�"����"�"�/�5�;�G�H�H�H�Z�R�Z�]�f�s�������s�f�Z�Z�Z�Z�Z�Z�Z�Z�	���������	��"�.�;�>�E�G�;�:�.�"��	�������������(�4�<�4�(�'����������������������������
����#���
�����������������������������������������������}�|�����������нݽ�ݽԽѽƽ��������(�����������4�A�M�]�a�c�a�Z�M�A�(�5�1�/�1�5�<�A�N�Z�g�q�s�y�w�s�i�g�Z�N�5�������������ʾ׾����׾ʾ�����������¿·²±²¿������������¿¿¿¿¿¿¿¿����ùïãàìðþ�����������	�������ҿݿӿۿݿ���������	�������ݿݿݿ��/�#��
�	�
���"�#�/�<�H�I�S�P�I�H�<�/�������������������ùĹϹչعϹù������������������ʼϼּ��������ּܼʼ����~�v�r�o�e�d�e�r�~�������������������~�~����������'�-�@�M�f�m�k�]�J�@�4�'�������$�0�V�l�~ǁ�{�o�V�=�$������ллû����ûлܻ�����ܻллллллܻԻػܻ����� ��'�4�>�?�0�������ܽ��y�l�Y�T�`�l�}�������������������������n�m�l�n�zÇÈÇ�|�z�n�n�n�n�n�n�n�n�n�nÓÍ��ÂÃÊÓìù��������������ùàÓĳıĦĚęęĚģĦĳĻĿ������������ĿĳĿļĿĹĹĿ����������� ��������������Ŀ C t 1 J ` 8 > + ] , 6 P ( � $ { 6 Z V ' - T p s . 0 J p S Y W M 7 ? U Z 1 x Q X C  y B 9 F Q $ j = $ A  T > r   $ ' 8 a 4 e 4 V q a � ; ; q V S - + F  �  _  �  R  u  �  
  {  �  �  �  �  J  >  �     �  Z    �  =    �  �  �  �  �  R  o  @  m  �  T  �  Y  >  �  �  A  �  A  �  �  }  �  E  �  �  �  �    p  �  I  Y  T  8  �  o  �  Z  �  �  L  �  �  �      �    �  2  �  u  �<�9X<u;ě�;��
;�`B%   ��o%   �ě��o�T���e`B�\)��C��ě����
��1��w�ě������+��1��j�o�H�9�+�P�`�'�h�0 Žixս49X�#�
��P�P�`�8Q�,1�8Q�'aG��}󶽏\)�P�`�0 ŽL�ͽ@���vɽ�%�P�`�8Q�u�H�9��q���y�#�]/��vɽ�`B���P��%��%��vɽ������-���罓t����w��1��F��{�ě���vɽ�����;d��9X��v�B�B�BHgB�NB1�nB!~rB&�BR�BAB`AB �~BY�B�SB��BeNB.z�B	�B'�BBoB��B�B�jB��A���A��BI
B
H�B�B+�B	�B�kB{�B �MBcBC~B	�CBB�cBB�B /B!3^B#SaB"Q�B
�RBLdBx B�GB&B%k�B(�B dB~�B�B�B �(BW�B&HTB&öBr]B��B
�B�pBKB	�B�B-l�B�B*��B� BbB�2B��B.�B	��B
DB��B�SB��B?�B�B2�B!�B&��B^_B>�B@�B!=�B�B�\B�=BB�B.�B	�YB��Bz�B�aB��B�B��A��A�~�B?�B
F�B@B, dB>�B>�B?zB �;B�YB?�B	��B/�B�B�wB ~<B ��B#��B"EB
��B4B?`BA�B��B$�9B(��B DmB�oB��B<B@6B=B&=�B&�%B�yBb�B	��B�dB�B��Bx�B-?�B~B*@FB�:B@�B>B��B3B	��B
/PB�gA	1B	�A��A�AJ�EA��A ��A��PAzj[A�/�Af"A��A�w�Ak9�AH<�AdFA���A��A�sA���A$�AL��AA��A��%A���A��RA�y�=�1~@��DA�(�A�Z�A�S'B�[A���A���AZߨA]�>A�I�A�OrA�'\@���@e�?��FA��A�;%A���C���>T��A��tA���A��AB A^3A2��A��>A�ǵA"�<A8��A��qAQI�A�HoA��8A�?+A»>8=VA \�@ 4@˾&B
ϒ@���@��AL�A�amA̅�A�^�A��A�B	�yA��A²lAK mA�sNA �A��Az��A�v�Ag�A�A�Aj�aAG.�Ab�kA���A�t�A�QA�z�A$&AK]A@�A�X,A���A�o�A��C��-@��}A���AˁNA�)BB�A�f?A��AZ��A[��A��Aͅ�A�sM@��/@\M�?�[0A�hA���A��;C���>M�1A��OA��A���AB��A]CA2�A�t�A�?�A!�A8��A���AQ�A���AτA~��A�>@�bA�@��@��B�@��Z@�p�A��AȀ�A͜�A�h�A�                     	            
                           
                                                   
   
            !         
      5                  &            +   >      
      "                     -               !                                 +            #                                    )      3            !                                 #                                                !            %                     %                                                '            #                                    )                  !                                                                                                                  %                     N��N��N���N$,�N1&lNrwhN�i&Nh��O���N��RO;��N��O��N=��N���N��NළN�<(N�N�j�O_��M��+NO,�P*8N���O�?3N�0wNm��O�9O���N��7O!`N��CO#�N���O�N���N(�O5E�O���OS}dN�6�NV�}O9~"O;�O
�Oln�O?�.N�C2N�/�NQ�O>KN�ѤN�x�N j�Oa�hO�֮O.N>N�iN4�kO�l�N�'�NǤhNW]�N��oN���OBVO�R�N���OB��O)�M�\�O��-O+8/Oϓ  �  �  l  +  p  j  �  H  �  �  �  �  4  �  {  �  �  �  �  )    q  P  �  �  �  f  �    K  3  �  K  e    u  �  e  x  �  �  �  �  V  �  �    �  �  �  �    6  ^  �  �  H  �    n  H  �  �    l  �  �  �  �  ;  �  u  �  C  e  �=+<�t�<D��<t�<t�;�`B;�`B;ě�:�o�o��o��`B�D���T���u�u��o���
��C���C����
��t���t���1��1��9X��h��9X���ͼ�`B��`B���������\)�\)�t���P��w��w�L�ͽ,1�#�
�''H�9�'''0 Ž0 ŽL�ͽD���P�`�L�ͽixս���Y��]/�q����o�}󶽇+��O߽�+��hs��hs�����������㽝�-���-���-���w��������������~{��������//<HSQH<7///////////:<HUUXZUH<<:::::::::%)6=BCB@<6)%%%%%%%%%OOY\houyuh`\VOOOOOOO��������������������3<IUbkcbZUMI<:853333����������������������5:EN[gpqog[5��������������������������������������������������������������FKQUa���������zaUPHF()6BCNOROJB6*)((((((����������������
***
���fgt�������~tgc]cffff#/1<HINH=<//#���������������������������T[_ht�����yth[XQQRQT����������������������������������������/7;?HPRPNMKHC;/,)((/ETaz����������zaTGAE#))5BIGB=<51)%######T[t�����������tgeZVT������������������������������������������


������������������� #/<AHRHH</'#      ������������������}�chhmt�������thcccccc #&./<>@AC?;1/# ^gt�������tjg\^^^^^^��������������������
#/28/#
	
#&#!
								wz~���������|zsnqrtw�������������������������
���������������������������������������������������������������������������������������
##)./253/#
	AHNUansxyzzsnaUOH@?A��
#0<IPQNI<90#
�dno{�����{tneddddddd��������������������#%"���� 
���������� ���{���������������{{{{ltz~�����tsmllllllll!#*0<IUX]^_IF<0#!,06IUbfjjfbUI<0+('(,����������DHPU[`a_VUHE>@DDDDDDnt~�����xtllnnnnnnnn����������������������������������NNS[[[gotvwtng[VPNNNX[hnsjh[TRXXXXXXXXXX�����������������vz�����������zzzusvv�������������������������������������������������������������������	
	���������������������������()57AA54)(((((((((((I[gt���������t][MIFIggmst�����������tjgg���������������������ּμּټ�����������ּּּּּּּ��$�$��!�$�0�5�3�0�$�$�$�$�$�$�$�$�$�$�$�;�8�1�:�;�H�T�`�`�U�T�H�;�;�;�;�;�;�;�;�/�(�#�#�!�#�$�/�<�E�@�<�/�/�/�/�/�/�/�/�����������������������������������������H�E�H�N�U�`�a�c�n�p�t�n�a�U�H�H�H�H�H�H�ʼ������żʼּ׼�������ּʼʼʼ�¦¬¬§¦�ѿʿʿ����������������ѿݿ��������ݿ��T�O�I�J�T�a�m�r�t�m�a�[�T�T�T�T�T�T�T�T�;�9�3�4�4�6�;�>�G�T�`�n�u�o�d�`�\�T�G�;�������������������������������������������������������������������������������˿`�_�X�`�`�m�o�y�~�y�t�u�m�j�`�`�`�`�`�`������}�������������������������������T�Q�G�E�;�3�.�*�'�.�;�<�D�G�N�T�Z�[�U�T��������������������	���	������������¿·³½¿��������������������������¿¿ŔŒŋōŒŔşŠŭŹ������ŹŵŭūŠśŔ���������������������������������������������������������Ľͽݽ���ݽнĽ����������������������������������������������M�E�M�Z�d�f�s�v���������s�f�Z�M�M�M�M���������q�s����������������������������������������(�7�?�I�W�Z�U�5��������������������������������������������������������������)�4�4�.� ������������������ùϹֹ���ܹϹƹù��������F�<�:�8�:�F�S�_�f�l�o�l�_�S�F�F�F�F�F�F��	������������������������	�����àÓÎÇ�z�a�a�p�yÇÓÝãìñùûøìà�H�D�B�C�H�O�U�a�a�b�f�a�_�U�H�H�H�H�H�H���������������������������������������¿³³¿���������������������������N�F�B�5�4�1�5�B�N�[�g�t�t�g�[�N������������	�������	���������������	����������"�.�;�=�@�<�;�.�)�"������ �(�)�6�>�=�6�1�)�!��������ÿùðóù�������������������������������	�����	��"�/�;�D�H�T�V�T�H�;�/�"������x�f�[�]�j�l�x���������������������������������-�2�:�@�A�:�3�-�$�!���L�L�H�F�L�Y�e�r�u�v�r�r�e�Y�L�L�L�L�L�LŹŹűűŹ������������źŹŹŹŹŹŹŹŹ�������������� �*�6�=�C�D�C�6�+��ŭŪŭŭŸŹ����������������������ŹŭŭE�E�E�E�E�E�E�E�FFFF$F)F$FFFFE�E򹪹��������������ùϹܹ�����޹ù��������������������������������������������g�[�Z�P�R�Z�g�r�s�|�������s�g�g�g�g�g�g�H�L�N�M�H�;�/�"����"�"�/�5�;�G�H�H�H�Z�R�Z�]�f�s�������s�f�Z�Z�Z�Z�Z�Z�Z�Z�	�����������	��"�.�:�;�A�A�;�5�.�"��	��������������(�4�9�4�(�%����������������������
���� ���
����������������������������������������������������������������������ҽнͽ˽�����������������4�A�M�T�Z�\�\�V�M�A�4�(��5�1�/�1�5�<�A�N�Z�g�q�s�y�w�s�i�g�Z�N�5�������������ʾ׾����׾ʾ�����������¿·²±²¿������������¿¿¿¿¿¿¿¿ùôèìõ�������������������������ù�ݿӿۿݿ���������	�������ݿݿݿ��/�.�#�����#�(�/�<�C�H�N�K�H�<�2�/�/���������ùϹҹֹϹù������������������������������ʼϼּ��������ּܼʼ����~�v�r�o�e�d�e�r�~�������������������~�~����������'�-�@�M�f�m�k�]�J�@�4�'�������$�0�V�l�~ǁ�{�o�V�=�$������ллû����ûлܻ�����ܻллллллܻ׻ٻܻ�������'�4�=�>�/�'����ܽ��y�l�]�_�l�y���������������������������n�m�l�n�zÇÈÇ�|�z�n�n�n�n�n�n�n�n�n�nÓÍ��ÂÃÊÓìù��������������ùàÓĳıĦĚęęĚģĦĳĻĿ������������Ŀĳ����ĿĺĺĿ���������������������������� C t 1 J ` 8 > + a , 6 P  o ( { 6 \ V ' 0 T p s . : * p S Y W O 7 ? U W ? x Q V D  % B 9 F L $ j = $ A  S 1 r   ' 8 a / e " = q a � ; ; m O S - + .  �  _  �  R  u  �  
  {  �  �  �  �  �  �  �     �  0    �  �    �  �  �  �  h  R  o  @  m  �  T  �  Y  �  U  �  A  �  ,  �  �  }  �  E  ;  �  �  �    p  �  %    T  �    o  �  Z  t  �  �  Q  �  �      �  �  k  2  �  u  P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  >P  �  �  �  �  �  �  �  �  �  �  {  u  o  l  h  a    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  M  #  �  �  �  l  g  c  _  W  O  F  ;  -         �  �  �  �  �  �  k  P  +  0  5  :  ?  C  B  A  @  ?  =  :  6  3  /  '        �  p  n  l  k  i  g  e  c  a  _  W  J  <  /  !       �   �   �  j  U  @  -    
  �  �  �  �  �  y  X  8    �  �  �  i  (  �  �  �  �  �  �  y  n  a  S  A  +    �  �  �  �  f  ;    H  ;  .  !      �  �  �  �  �  j  M  2      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  2    �  �  ~  F  �  �  �  �  �  �  �  }  t  i  _  T  J  A  8  -         �  �  �  �  �  �  �  �  �  �  �  z  v  u  t  t  p  f  [  K  ;  (    �  �  �  |  s  i  e  e  f  j  n  r  r  r  r  q  q  p  p  p  �    !  0  4  %    �  �           �  �  �  w    �    �  �  �  �  �  �            �  �  �  �  �  �  �  �  �  z  z  z  t  k  W  A  '    �  �  �  �  i  F  $    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  |  z  j  W  D  1  �    z  t  n  d  Z  O  G  @  9  2  (        �  �  �  �  �  �  �  �  �  �  �  }  [  :    �  �  �  U    �  9  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  i  M  1    �  �  )        �  �  �  �  �  �  �  �  �  m  U  :    �  �  �  �  	            	     �  �  �  �  \  *  �  �  J  	   �  q  m  i  f  b  _  [  R  F  :  .  "    	   �   �   �   �   �   �  P  L  H  D  >  ,    	  �  �  �  �  �  ~  c  I  (     �   �  �  ~  o  b  W  M  C  9  0  %    �  �  �  ^    �  v     �  �  �  �  f  G    �  �    <  E  8  !    �  �  h    �  �  �  �  �  �  �  �  �  �  �  �  �  t  _  F  (  �  �  G  �    �  �    F  \  e  `  K  /    �  �  �  e  !  �  �  D  �  �  �  �  �  �  �  m  B    �  �  �  �  `  %  �  \  �  w   �   �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  K  G  :  '    �  �  �  �  �  |  Q    �  �  X    �  y  '  3  .  $      �  �  �  �  a  4  �  �  d  
  �  I  �  �  z  �  �  �  �  �  �  �  }  l  [  G  ,    �  �  �  e  7    �  K  A  6  -  "      �  �  �  �  �  �  g  A    �  l   �   /  e  Y  L  >  ,          �  �  �  �  �  �  �  �  �  w  ^      �  �  �  �  l  C    �  �  H  �  �  0  �  m  �  V   �  n  r  t  u  t  q  j  _  P  ;    �  �  r  (  �  �  *   �   q  g  y  �  �  �  �  �  �  ~  n  X  @  $    �  �  �  p  ]  I  e  W  I  <  .  0  7  F  Y  [  Q  E  8  *      �  �  �  �  x  i  Z  K  <  /  &          �  �  �  �    $  F  i  �  �  �  �  �  �  �  �  m  P  N    �  �  �  z  z  k  F    �  �  �  �  �  ~  l  \  H  2    �  �  �  s  ,  �  _  �  �  >  �  �  �  �  �  �  �  �  �  �  �  �  v  P  !  �  �    �   �  P  J  T  u  �  �  }  l  Z  E  ,    �  �  �  \  /  7  1  !  V  K  @  5  *        �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  s  j  e  `  \  Y  U  R  O  K  G  B  ;  -    �  �  �  �  �  �  �  v  h  Y  I  9  '    �  �  �  �  �  O  
   �  
�  
�        
�  
�  
�  
�  
;  	�  	�  	  �  *  �    %  )    �  �  �  �  k  H  $  �  �  �  Z    �  �  K  �  �  F  �  {  �  �  �  �  �  �  �  �    q  o  y  z  t  m  g  n  x  r  j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  d  M  7     
  �  �  �  �  �  x  e  N  1    �  �  �  a  4  
  �  �    Q          �  �  �  �  �  �  �  �  �  �  _  <    �  �  r  2  5  6  6  2  *       �  �  �  \    �  s    �  W    �  W  [  X  J  8  %    �  �  �  �  �  �  �  u  Z  1  �  �  M  �  �  �  �  �  �  �  �  �  �  c  3    �  �  �  �  p  F    �  �  �  �  �  �  �  �  �  �  �  �  �  {  e  X  L  A  5  *  �  -  A  G  =  .      �  �  �  q  ?     �  `  �    �   �  #  �  �  �  �  �  �  �  �  X     �  �  ,  �  H  �    %  �    �  �  �  �  �  �  v  W  2  	  �  �  z  <  �  �  <  �    n  e  \  S  K  C  ;  1  '              	    �    !  H  >  3  )          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    o  Z  @    �  �    D    �  q    �  �  �  m  7  
        �  �  �  f  ,  �  �  �  �  �  �  �  �                   �  �  �  �  �  j  B    �  �  �  �  �    >  b  l  i  f  a  Z  L  7    �  �  z  >     �  �  J  �  �  ~  l  Y  F  4  !    �  �  �  �  �  �  o  M   �   �   ]  �  �  �  �  �  �  p  [  G  /    �  �  �  �  �  �  �  �  �  �  f  M  <  M  J  9    �  �  �  �  h  4  �  �  �  U     �  �  �  �  U    �  �  w  >  	  �  �  8  �  S  �    '  �  �  ;  6  1  +  $        �  �  �  �  _  -  �  �  �  j  o  z  �  �  �  �  �  �  �  �  �  �  i  )  �  �  D  �  �  d  g  �  s  u  t  r  m  f  ^  S  B  $  �  �  �  n  0  �  �  N  �  �  �  �    u  l  b  Y  O  F  <  +    �  �  �  �  �  y  _  F  C  4    �  �  �  Y  +     �  �  s  C    �  �  3  �  @  t  e  Y  L  5      �  �  �  �  n  L  '    �  �  �  x  6   �  ]  �  �  �  t  `  L  <  .      �  �  �  |  F    �  �  j