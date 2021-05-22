CDF       
      obs    U   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�     T  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�s�   max       Pk��     T      effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ȴ9   max       =,1     T   T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @Fe�Q�     H  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vz�\(��     H  .�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  <8   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�@         T  <�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <�     T  >8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B/�|     T  ?�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�r�   max       B/�     T  @�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >g�w   max       C���     T  B4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Bx�   max       C��     T  C�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i     T  D�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     T  F0   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     T  G�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�s�   max       Pk��     T  H�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?�@N���     T  J,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ȴ9   max       =,1     T  K�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F.z�G�     H  L�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vz�\(��     H  Z   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q�           �  gd   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�@         T  h   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @t   max         @t     T  id   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�@N���U   max       ?�V�u     �  j�                                                #            	         #         3   !                           9         -            	                           	   "      d   >                     ,      	   i         
      	   	                     ?   +               Ns�GN�N���O}+zNT�N?��NEN)�N�R:O���N�*�N��]O.lOa�N-T�PO��@N��ZN/��N�&nOa�O;GuO��OҮ�N!<�P�iO�gN�zTO�nN���N��O�ODrRN�d�ORL�Pk��N�r�Nq�~P��O��IO%�O��N�K�N�I�N�3�N梫N^w�Ne�bM��N�e?N��N�[AOr��M�s�P��O���N�q�O6�N�iO�RN���N�l�OQ�O�PN�D-O�9VN�iiN�)�N�O��NŗN��N��OU:�NؐqO' O4�SN!��P��O�A\OH0:N> NHkNs6�Nm	y=,1<�9X<�o<T��;o:�o%   ��o�D�����
���
�ě��49X�D���T���T���e`B�e`B�u��o��o��C���C���C���C���C���j��j��j�ě��ě����ͼ��ͼ�`B��`B��`B��h��h��h���o�C��t��t��t��t���P��P��P��w�#�
�0 Ž0 Ž0 Ž8Q�8Q�8Q�8Q�<j�<j�<j�<j�@��@��D���H�9�P�`�T���T���T���Y��]/�aG��aG��e`B�e`B�ixս�o��o��hs���P�������9X�ȴ9������������������������������������������������������)5@:EB5)�������������������������������������?BHNYVQNB>??????????���		�����������������
�������msz�����������zmgafm��������������������269BOPZ[a_[OIB962222��
#%%##!
����#<IUbkkfbUMI<7/(#��

��� ���������������������������������	������������������������������
"
�()-5BCDDCB:5)  #((((��������������������	)3/31165)	MP[g������������t[RMHN[g�����������teXNH�������������������*6CO}yh\6*����;?HTamz����zZTH;85;`ajmntvuwusmfa`]^^``����� 

���������������������������KN[gqtttsg[NIFKKKKKKty��������������vrqt#$&-/<HUadda^VUH</##-/<HHOQHC<<///------�����������������������#<bn{����{U<0
����������������������
#(/3/#"
����	(--�������|�����������������|��	���������������������������BN[gmrig[NGB>BBBBBBB����������������������


	��������W[^htu|�|wthf`[YVRWWNN[gtxwtg[NKNNNNNNNN`hiitw~|the_bf``````��������������������56BOU[[[WOIFBB?;6555����	���������������������)/5BDGJLNNEB5)& ������������������������������������������������[[dhtw����th[X[[[[[[��������������������� EN[gt����������rc[FEGOQ[hioihb[ONEGGGGGG[hqt��������tjha[[[[")5;==;82)
)BNQ]^[TK?75/)%�����������������������#/336:/)
����wz|��������������zww������������������������������������������
!#+0:0/,#
	 ���� 
#&%#
 ���������������������^anuqnba\V^^^^^^^^^^nz���������{znlifghn��������������������wz������������}zxxwwHIU_bfkmxrnbUIFC>>DH�������������������������������������}����������������������� "" 	����
+/;<HNOH<://++++++++,/<BB=<6/')),,,,,,,,���������������������ݽӽֽݽ���� ��������ݽݽݽݽݽ�¤¦²¿����¿½²±¦�
�����������
���#�.�/�9�/�#���
�
�
���������������)�5�+�(�3�;�2�)�����������������������������������������׾���������������������������������������������������������������������������������
�	��	�� �"�/�5�/�"���������n�j�a�U�U�S�U�\�a�n�p�y�zÄ�z�x�n�n�n�n�����ܿڿݿ���(�A�N�Z�[�I�F�F�A�5�������������� ��	��� �*�/�+�*��������a�Z�U�H�R�U�a�d�n�zÃÂ�z�x�n�h�a�a�a�a�Y�M�@�4�*�4�=�@�M�Y�f�r�������w�r�f�Y���������������ʼּܼ�����ּʼ�����ŭŬŦŧŭŹſ����������Źůŭŭŭŭŭŭ���y�h�W�O�M�N�T�e�m���������ɿҿϿƿ���������������������������������������������������*�6�?�6�2�6�C�C�C�6�*���$�����$�0�8�3�0�$�$�$�$�$�$�$�$�$�$�"������"�/�;�=�H�N�L�H�;�/�"�"�"�"��������w�y��������������ʾʾ��������������������(�5�A�K�L�G�A�6�5�(����	���������������	��"�+�0�/���	���ݿ�׿Ͽпؿݿ������"�!�������Z�Q�N�A�@�A�N�Z�g�i�g�g�Z�Z�Z�Z�Z�Z�Z�Z�.����������"�.�>�E�R�R�\�Z�T�G�.����ƻƬƫƭƳ����������������
������čćā�āčĚĦĳĿ��������ĿĳĦĚčč���������������������ƿпѿӿѿ̿ȿĿ����$�"�������$�0�1�8�9�5�0�%�$�$�$�$���������������������������������������������������������)�4�9�%������������"��	������������	���"�.�0�4�1�.�"�"�
���
�
��#�/�/�/�/�$�#��
�
�
�
�
�
�������������ɺֺ������������ֺɺ������������v�W�Q�S�]��������������������ìàìïù����������������ùìììììì�����(�-�4�?�A�A�A�A�6�8�4�(�������������˼ּ���!�.�5�7�4�.�%���㼽�����������������
��6�<�I�U�I�0�#������.�&�&�.�;�G�K�T�`�m�y�|�|�w�m�`�T�G�;�.�������������'�4�@�N�R�@�4�/�������������������
����������������/�-�/�0�/�.�/�2�;�G�H�L�T�]�`�X�T�H�;�/��������)�6�7�6�6�2�)�������f�d�f�l�q�r���������������������r�f�fŠŠśŜŞŠŭůŴŴűŭŠŠŠŠŠŠŠŠ�������������	�����	�����������������	����	�	�
������	�	�	�	�	�	�	�	�;�:�:�7�;�G�G�H�T�]�`�m�n�m�`�T�J�G�;�;ĚĘčăĉčĚĦĳĿ������ĿĳĦĚĚĚĚ�������������������������������������������������������*�6�C�O�R�P�G�C�6�����m�m�j�m�m�s�z�z�}�z�m�m�m�m�m�m�m�m�m�m�j�b�d�n�������ûۻ������ػû����x�j�������	��'�4�@�M�X�^�_�^�Y�M�@�4�'������	����'�'�,�-�0�'��������������������"�'�-�1�3�6�3�'��B�:�6�)��)�6�B�L�O�[�^�h�n�t�u�t�h�O�B�t�k�j�n�xčęĺĿ�������������Ŀİā�t�ù����������ùϹӹܹ�ݹܹϹùùùùùû��{�x�x�w�x�x��������������������������������ÿùðìù���������������������H�<�3�0�7�<�H�U�`�n�zÇÓÛÓÑÇ�r�U�H�~�u�{¤ D�D�D�D�D�D�D�D�EEEEECEJEBE7E)EED��ѿǿĿ����ÿĿѿҿݿ�����ݿܿտѿ��I�I�@�I�U�U�U�b�n�{ŇŉŇ�~�{�n�b�U�I�I�#���
��������������
��#�(�-�0�0�/�#���������������������ûлջһѻлû������z�u�m�a�[�\�a�g�m�z�|���������������z�z���u�s�k�r�s��������������������������������������������������������������������H�@�B�H�T�a�m�x�z�~�����������z�m�a�T�H������
�������������������������뾘���������������¾ʾ׾۾۾׾ʾþ��������F�C�G�S�U�_�l�x�����������������x�_�S�F�ܹڹܹ�����������ܹܹܹܹܹܹܹܺ���3�L�e���������ĺƺ������r�e�L�'��ʼż������ɼ���������������ּʽ���!�.�:�S�`�l�y�������y�a�S�G�.�!��������������Ľ̽ǽĽ�������������������FFFFFFF$F.F&F$FFFFFFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������
�����
�������������������� M @ : d f ; J ~ 9 8 & L * I I  # E ; R X R : W N \ ) s 9 c , Y ' ; / h b t R Y Y [ 8 7 > K e v � G E + U v 5 # O , f m 4 H T 3 ( 9 ` M @ = C 3 ] / # 4 > i q I y f o g $    �      F  _  N  &  P  �  ^    �  |  �  [  2  �  �  C  �  L  �  �  \  B  :  �  9  :  -  �  U  �  �  �  �  �  �  �    �  s  �    �    �  �  �  �    �  $  (  �  S  �  G  @  �  �  )  �  \  �          ]  �  �  %  �  �  c  �  l  �  4  ,  \    |  m<�<49X;D����o��o�D���o��o�49X�\)�u����C��\)����@��C���1��9X�ě�����h�P�`�8Q켣�
��7L�e`B�t�����P�#�
�]/�D���+�T����1�\)��󶽗�P�aG��<j�}�8Q�Y��L�ͽq����w�,1�#�
�0 Žu�P�`���-�<j�z��"ѽixսq����\)�����O߽�o��Q콓t��ixվ��ixս�%�}󶽁%�}󶽁%�m�h��+������C������t��J��l���^5���w��9X������hB)�B�ZB��B��B�BM�B'A��B{B ��BFhB��B$b4B&��B��B+�B��B�yBB;�B��B8�B	�B
�B7�B/�|A��A�cJB5�Bc�B��B��B6B�HB"F|B&�B�&BS�B-�5B�B+�B �YB��B��B��B�pB��B��B `�BjB��B29BE�B��B�BϿB��B!Y�B�LB	�QB/$BnBdBa�B�fB��BAB�7B�GB$�)B�B�?Bw�B��B�	B�B'X!BߜBˋB��B�B]�B��Br�B$B?-B��B��BBpB?lB2 A�r�BAfB ��B69B�B$@QB&��BuB+=�B�B?�B; B=�BBS�B	��B
��B?eB/�A� `A��5BBLB�B
��B�wB��B">�B&}`B?�B.B-�B��B@�B ��B�B�QB�qB��B�SB��B U�B�@B�jBjhB?�B�]B@YB��B+B!K�B�lB	�rB?B��B�7B��B��BP�BD�B�sB��B$�DB��B��BS�B��B��BùB&�:B�ZB�	B[�BF{B=�B=GBB]B=�A-t�A�L�A�Z�A�+A�m�AI&AI�A�[A�WA�N�A�R�A���@��@�ueA�D�Ap�%A��)A���B	�XA��]AJ}GA�mgAZ�-A�(A��A^XfB��A��HAv�bB	�zA��pA�RA\��A���@9�xA���A�p�A8E�AjgA�"Af�I@ɘ�A0A�O�AՖ<@�Z!A��rA���A���Af�A�(A�лA�hA�"�@�5'@��.?��U?w��A�`8A�[>g�w@��DAБ�A�?�A�a:C�S_A|�A�ŖA�6}@���A�p8A���A�U�A��2A��jAM3U@�(8?&��?�5�A�wA%UA$3�C���C�.pA�r�A-cA��A�EA��A���AH6;AH�A�4KA�]A��1A���Aƀ�@ݴ�@��A��App�A���A�M�B	�-A�q AJ�RA�x+AZ��A��~A��A\��B�A��Aw+B	ےA��A�M�A[�A���@4�A��fA�~�A7�A��A�Aj��@���A1\A��$AՃ�@靳A�;A�~:A��Af��A���A��A��NA���@��@�%?��?p�3A�|:A���>Bx�@���A���A�K�A�{�C�O�Az��A�w�A�}�@��A�x A��'A�w�A��~A�P�AK�c@�O|?-��@+A�UAl�A#mC��C�(oA�~�                           	                     #            	         #         3   "      	                     9         .            
                           	   #      e   ?                     ,      
   i               
   
               	   	   @   +                           !                  %                  '                     !   %      +                  '            9         +   '                                             '               )                  %                                       /                                                                  %                                                '            9         #                                                               %                                                         +                  Ns�GN��LN�r�O)��NT�N?��NEN)�N}d-Oƭ�N�*�NB�N���N�N�N-T�O�]<O[�vNx/NN/��N�&nOa�N�H�O#�(O��	N!<�Op�hO�gN�zTO�nN���N��O�O8`N�d�O��Pk��N�r�Nq�~O��\O�aLO%�N�T�N�K�N�D@N�3�N梫N^w�Ne�bM��N�e?N�(<N�[AN�uUM�s�O�IcOe!Nf�4N��,N�TOԆoN;�-N�l�O�hO���N�D-OJ��N�iiN�)�N�O!NŗN��N��OU:�N��~O' O4�SN!��Ph&O�A\OH0:N> NHkNs6�Nm	y  �  �    �    �  �  �  �  �  r  �  h  *  n  �  f  r  �    �     �  �  j  \    �  �  �  	  �  �    n    :  �  �  �  �  �  �  +  [  �    �  D  �  l  �  �  �  �  
�  �  �  �  �  �  2      f  N  �  �    >    �  H  6    �    6  	q    �  w  �  n  �=,1<���<e`B<#�
;o:�o%   ��o��o�#�
���
�D����o����T����o��9X�u�u��o��o��1��h���
��C���w��j��j��j�ě��ě����ͼ�����`B�o��`B��h��h��P�+�o��P�t���P�t��t���P��P��P��w�'0 Že`B�0 Ž��w�m�h�@��D���D���H�9�Y��<j�aG��H�9�D����1�P�`�T���T���Y��Y��]/�aG��aG��q���e`B�ixս�o��7L��hs���P�������9X�ȴ9���������������������������������������������������������� )565)�������������������������������������?BHNYVQNB>??????????���		�������������������������mz������������zmhgim��������������������BBCOU[][OB@:BBBBBBBB��

!
��������.0<IUbbdb_UTI<<500..��

��� �������������������������������������������������������������������
"
�()-5BCDDCB:5)  #((((��������������������)+)&)))*)
	Y[gmt��������tog\[WYT\g������������tg\XT�����������������
*6CKRTQJC6*;?HTamz����zZTH;85;`ajmntvuwusmfa`]^^``����� 

���������������������������KN[gqtttsg[NIFKKKKKKty��������������vrqt./<HU`aba]UTH<3/#$'.-/<HHOQHC<<///------�����������������������#<bn{����{U<0
����������������������
#(/3/#"
����#'#�������������������������������	���������������������������BN[gmrig[NGB>BBBBBBB����������������������


	��������W[^htu|�|wthf`[YVRWWNN[gtxwtg[NKNNNNNNNN`hiitw~|the_bf``````��������������������56BOU[[[WOIFBB?;6555����
���������������������))5=BDFGEB50)&%%))))�����������������������������������������	��������Y[\ghpt���th[YYYYYY���������������������� HN[gt��������te[SLHHKOW[ahjhd[OKKKKKKKKK[hqt��������tjha[[[[')59:9650)"!)BN]][RIB<5)&����������������������
!"!
�������wz|��������������zww������������������������������������������ 
 #(/-*#
����� 
#&%#
 ���������������������^anuqnba\V^^^^^^^^^^nz���������{znlifghn��������������������wz������������}zxxwwHIU_bfkmxrnbUIFC>>DH��������������������������������������~���������������������� "" 	����
+/;<HNOH<://++++++++,/<BB=<6/')),,,,,,,,���������������������ݽӽֽݽ���� ��������ݽݽݽݽݽ�¦²¹¸²ª¦���
� ��������
���#�,�/�4�/�#������������������#�'�+�6�,�)�����������������������������������������׾���������������������������������������������������������������������������������
�	��	�� �"�/�5�/�"���������n�l�a�W�U�T�U�`�a�g�n�v�zÁ�z�s�n�n�n�n������������A�M�Q�L�D�?�=�5�(�������������� ��	��� �*�/�+�*��������U�T�U�Z�a�l�n�r�u�q�n�a�U�U�U�U�U�U�U�U�M�K�@�@�9�@�M�Y�f�r�y�|�s�r�f�Y�M�M�M�M���������������ʼмּ�����ּʼǼ���ŭŬŦŧŭŹſ����������Źůŭŭŭŭŭŭ���y�k�Z�R�P�W�`�m���������Ŀοʿ�������������������������������������������������������*�6�/�*�"���������$�����$�0�8�3�0�$�$�$�$�$�$�$�$�$�$�"������"�/�;�=�H�N�L�H�;�/�"�"�"�"��������w�y��������������ʾʾ��������������� ������(�5�:�=�5�-�(�������������������	���"�$�%�"����	�������ٿѿӿܿݿ������� �������Z�Q�N�A�@�A�N�Z�g�i�g�g�Z�Z�Z�Z�Z�Z�Z�Z�.�"��	����������	��"�.�4�:�>�?�;�5�.����ƻƬƫƭƳ����������������
������čćā�āčĚĦĳĿ��������ĿĳĦĚčč���������������������ƿпѿӿѿ̿ȿĿ����$�"�������$�0�1�8�9�5�0�%�$�$�$�$���������������������������������������������������������)�4�9�%������������������������	���"�.�.�3�1�.�-�"��	���
���
�
��#�/�/�/�/�$�#��
�
�
�
�
�
�ɺ��������������ɺֺ��������ֺ������������v�W�Q�S�]��������������������ìàìïù����������������ùìììììì�����(�-�4�?�A�A�A�A�6�8�4�(�������������Ӽ���!�/�1�-�'������ּʼ������������������
��.�0�;�0�(�#������.�&�&�.�;�G�K�T�`�m�y�|�|�w�m�`�T�G�;�.���������'�4�@�K�M�N�M�@�4�)�'����������������
����������������;�2�/�3�;�H�T�\�_�W�T�H�;�;�;�;�;�;�;�;��������)�6�7�6�6�2�)�������f�d�f�l�q�r���������������������r�f�fŠŠśŜŞŠŭůŴŴűŭŠŠŠŠŠŠŠŠ�������������	�����	�����������������	����	�	�
������	�	�	�	�	�	�	�	�;�:�:�7�;�G�G�H�T�]�`�m�n�m�`�T�J�G�;�;ĚęčĆčĎĚĦĳĿ����ĿĽĳĦĚĚĚĚ�����������������������������������������������*�6�C�G�G�C�;�6�*������m�m�j�m�m�s�z�z�}�z�m�m�m�m�m�m�m�m�m�m���x�o�p�y���������ûлֻڻ׻ͻû��������'�!������'�4�@�M�S�[�[�Y�O�M�@�4�'�'���������$�'�+�,�-�'�'�'�'�'�'����������������'�'�*�(�'����B�>�6�)�&�)�6�B�O�[�h�l�r�h�[�O�B�B�B�B�t�n�l�q�{čĦĿ������������ĿĬĚčā�t�ù����������ùȹϹڹչϹùùùùùùùû��{�x�x�w�x�x����������������������������������ù÷ù������������� ���������H�A�<�7�3�9�H�U�]�n�zÃÇÒÎÇ�o�a�U�H�~�u�{¤ D�D�D�D�D�D�D�EEEEE*E/E*EEEED�D�ѿǿĿ����ÿĿѿҿݿ�����ݿܿտѿ��I�I�@�I�U�U�U�b�n�{ŇŉŇ�~�{�n�b�U�I�I�#���
��������������
��#�(�-�0�0�/�#���������������������ûлԻллû��������z�u�m�a�[�\�a�g�m�z�|���������������z�z���u�s�k�r�s��������������������������������������������������������������������H�@�B�H�T�a�m�x�z�~�����������z�m�a�T�H�����������������������
��������������������������¾ʾ׾۾۾׾ʾþ��������F�C�G�S�U�_�l�x�����������������x�_�S�F�ܹڹܹ�����������ܹܹܹܹܹܹܹܺ)��3�9�L�e���������ºź��������r�e�L�)�ʼż������ɼ���������������ּʽ���!�.�:�S�`�l�y�������y�a�S�G�.�!��������������Ľ̽ǽĽ�������������������FFFFFFF$F.F&F$FFFFFFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������
�����
�������������������� M + 6 Z f ; J ~ 8 5 & X * 0 I  & ( ; R X R 5 V N E ) s 9 c , Y % ; 8 h b t H F Y R 8 2 > K e v � G A + D v 0  V 0 f o ) H N * ( 2 ` M @ < C 3 ] / ) 4 > i i I y f o g $    �  �  �  �  _  N  &  P  �  �    V  �    [  �  �  �  C  �  L  �  c  �  B    �  9  :  -  �  U  �  �  [  �  �  �  �  X  �  '  �     �    �  �  �  �  �  �  �  (  �  �  �  �  �  �  N  )  s    �  �        /  �  �  %  �  �  c  �  l  
  4  ,  \    |  m  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  @t  �  �  �  �  �  �  �  �  �  \  $  �  �  0  �  b  �  v   �   {  �  �  �  �  �  �  �  �  �  �  x  \  B  (    �  �  �  �  �          
  �  �  �  �  �  �  m  L  &     �  �  �    \  �  �  �  �  �  �  �  �  �  �  �  p  U  H  >  2    �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  {  v  p  j  d  _  �  �  �  ~  o  `  M  ;  (      �  �  �  �  �  t  \  D  ,  �  �  �  �  �  {  v  q  l  g  ]  O  @  2  $       �   �   �  �  �  �  �  �  �  �  {  u  o  j  d  ^  R  <  '    �  �  �  �  �  �  �  �  �  �  �  �  p  Z  B  )    �  �  �  �  �  [  �  �  �  �  �  �  �  �  j  :    �  �  B  	  �  e  	  �  Z  r  g  \  T  L  @  3  !    �  �  �  �  �  �  V  	  �  [    M  q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  /  G  Y  b  g  h  e  \  L  9    �  �  �  {  +  �       �  �  �  �    !  )  )      �  �  �  �  �  z  =  �  C  �   �  n  h  c  ^  R  F  9  -         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  k  Z  C  $  �  �  �  ;  �   �  �    -  @  O  Y  a  e  d  ^  R  <    �  �  �  _    �  9  O  \  i  o  e  [  Q  F  <  1  '        �  �  �    0  Y  �  �  �  �  �  �  �  �  �  �  q  ]  E  .    �  �  �  �  �    {  w  q  i  `  R  D  1      �  �  �  �  ^  5     �   �  �  �  z  k  \  J  ;  -         �  �  �  �  f  ,  �  �  n  �    	                  �  �  �  �  �  p  S    �  �  <  _    �  �  �  �  �  �  ~  S  "  �  �  M  �    �   �  �  �  �  �  �  �  �  �  f  3  �  �  �  c  0    �    �   �  j  Y  I  8  '      �  �  �  �  �  �  p  Z  C  ,     �   �  �  �  �  �    A  R  Z  \  Y  E    �  �  u    �  (  g   �      �  �  �  �  �  �  y  E  
  �  x    �  H  �    �   �  �  �  �  �  �  |  ^  :  	  �  �  9  �  �  <  �  �  0  �  v  �  �  �  �  �  �  }  f  M  5      �  �  �  �  i  ;   �   �  �  �  �  �  �  �  �  �  �  �  t  \  C  )    �  �  �  �  }  	      �  �  �  �  �  �  �  t  A  �  �  Q  �  �  *  �  V  �  �  �  �  �  �  �  �  �  �  �  }  ^  .  �  �  X  7  �    �  �  �  �  �  l  R  3    �  �    K    �  �  e    �  C    �  �  �  �  �  �  �  �  �  �  �  v  k  `  T  J  @  7  -  :  a  k  n  k  c  U  ?  !  �  �  �  ]    �  �  T  &  A  �    �  �  �  }  D    �  }  8  �  �  �  8  �  �  c      �  :  2  +  #    �  �  �  �  �  �  [  +     �  �  �  }  a  F  �  �  �  �  �  �  �  �  �  �  |  k  Z  J  9  (       �   �  w  �  �  �  �  �  �  l  I  "  �  �  �  B  �  �  #  �  �   �  O  �  �  �  �  m  N  (    �  �  V    �  |  -  �  �  4  �  �  �  �  �  �  �  �  �  �  �  �  c  C     �  �  �  `  +  �  8  k  �  �  �  �  o  G    �  �  Z    �  �  \  "  �  �  �  �  �  �  u  b  M  7       �  �  �  �  X  (  �  �  k  !   �  �  #  "      �  �  �  �  �  {  V  -  �  �  E  �  h  �  s  [  G  6  &    	  �  �  �  �  �  �  �  t  ]  F  :  -      �  �  �  �  �  o  I    �  �  �  r  G  �  �  U    �  �  H          	      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    ~  }  |  s  d  V  H  8  '      D  :  /  %        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  a  V  K  ?  4  +  $        
     �   �   �  j  k  a  V  C  (  	  �  �  �  w  L    �  �  �  c  )  �  z  �  �  �    o  _  O  ?  0         �  �  �  �  �  �  �  �  n  �  �  �  �  �  �  �  �  �  �  v  )  �  D  �    n  �  q  �  �  �  �  �    z    �  �  �  �  �  �  �  �  �  �  �  �  
�  8  }  �  �  �  �  �  �  �  1  
�  
=  	�  		  5  $  �  i  @  
  
p  
�  
�  
�  
�  
k  
D  
  	�  	y  	  �  �  [  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  l  X  C  .     �  n    �  �  �  �  �  �  �  �  �  �  �  �  _  ;    �  �  {    �  ]  �  �  �  �  �  w  L    �  �  F  �  w  �  p  �  E  �  �  &  �  �  �  �  �  n  V  ;  !  �  �  �  n  R  9    �  T  �    F  h  �  �  �  �  �  �  �  �  �  U    �  P  �  G  �    h  2      �  �  �  �  Z  4    �  �  �  b  1  �  �  �    �  
�  
�  
�        
�  
�  
�  
p  
(  	�  	r  	  �    �  �    �  �  	  
    �  �  �  �  J    �  �  S    �  a  �  �    �  f  b  ^  V  N  A  3      �  �  �  �  h  @    �  �  9   �  �  �  �  �  "  B  K  H  3    �  �  3  �  K  �  �  
�  	h  ~  �  �  �  �  |  x  u  q  k  f  \  O  A  .    �  �  �  �  b  �  �  �  �  �  �  s  ]  C  )    �  �  �  �  �      *  7      �  �  �  �  �  �  �  �  �  �  i  >    �  �  �  �  �     2  :  *      �  �  �  �  �  p  X  G  8  ,  $  "            �  �  �  �  �  �  �  �  �  �  w  c  L  -    �  �  �  m  [  D  -    �  �  �  �  �  d  C    �  �  }  R  %   �  H  E  B  ?  <  :  7  .  #        �  �  �  �  �  �  �  v  6  /  $    �  �  �  �  �  x  V  0    �  �  �  �  �  �  h                 �  �  �  r  M  )    �  �  q  =    �  �  �  �  �  �  �  }  d  H  (    �  �  �  U  -    �  �  �                �  �  �  �  �  �  �  �  �  �  �  �  �  6  5  4  2  /  *        �  �  �  �  �  e  E  $    A  d  	l  	n  	R  	2  	  �  p    �  R  3    �  h    �  �  J  �   ^    �  �  g    
�  
e  	�  	�  	  �    �    �  �  �    5  �  �  �  �  �  �  w  r  h  S  8    �  �  �  y  ;  �  \  �  ?  w  u  s  q  o  m  l  j  h  f  d  b  a  \  Q  F  <  1  &    �  �  t  U  X  c  m  _  F  ,      �  �  �  �  �  �  �  �  n  <    �  �  a  *  �  �  z  F    �  �  �  q  R  3    �  �  �  �  �  �  ~  h  P  5    �  �  �  N    �  N  �  n  �