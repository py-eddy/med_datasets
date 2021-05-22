CDF       
      obs    P   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�     @  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�]�   max       P��     @  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       <D��     @   ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F�=p��
     �  !l   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�H    max       @vfz�G�     �  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @P�           �  :l   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��`         @  ;   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �hs   max       ;�o     @  <L   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�NK   max       B1m     @  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�eL   max       B0�     @  >�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =���   max       C��     @  @   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��L   max       C��     @  AL   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i     @  B�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     @  C�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     @  E   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�]�   max       P���     @  FL   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?�ѷX�     @  G�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <D��     @  H�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @FxQ��     �  J   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vfz�G�     �  V�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P�           �  c   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��         @  c�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�     @  d�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?�������        f,               :               7   9               	      D            X   $   !         	      	                  #      
                     /            i      (               E                  
   M      	   (                                             O�+�N���O>xN;�sP�KN/�NT��N���O���Pa:P/?Nd�O N'�N>��N��XN�IjP*��N�O�N� �OP��P��P��O��O\�O �N�ROe'�N���Oj@OG02N��_O�v�N��"O�L�O�?N-��N@�(N�L�N3
OM���N$.�O��UO�/�O���N
-aNfAP4AiN�O�ۖN %9O��O��O�O_O�ߏOA�O.��OT�6N$JN`�N�:�P=2HN��TO L1O���N���N�3�N��OH}8O���N���NtZ�O��\NgXyOj 6O��hN�2,N2��N�O�M�]�<D��<t�;�`B;�`B:�o��o�D���o�o�t��#�
�#�
�49X�49X�D���T���e`B�e`B�u��o��C����㼛�㼣�
���
���
��1��1��1��9X��j��j��j��j�ě����ͼ���������`B��`B��`B���o�+�+�t���������w��w�'''<j�<j�D���H�9�H�9�L�ͽP�`�Y��]/�]/�aG��aG��ixսu�y�#�y�#�y�#��+��t���t���-��-��E���E��Ƨ�Ƨ������
"($
�������

����������������������������������������az������������m[WUYa���������������������������������������./<HOUXYURH<2/......(6BW[hjssjeXOB6/*!6Ohu������sgO*GTaz��������zmaTD@AG���

���������EO[hqt���thb[XOLHEE����������������������
�������������IOR[dhtuwtoh[YOMIIII��

������������������������|{�����������������������������������������")10)�������5[tw���g^PB5
���������������������������������������)6AOS[[XOA6!����������������������			��������DOht���������th[OJDD����������������������������������������������� �����������9BEO[hijiihg[ROLCB99��������������������������������������������������������������������������������)6;76)?BHO[hjhd[OB????????//1;>=>BF;7/**+.////^agnzz�zqna]^^^^^^^^����������������������������������������!>N[gt����ztg[IB5' !uz}��������������ysu�������	���������������������������������������������:=HUaz��������yaUH?:5<AHIU\ZZUNI<6555555B[q����������g[N<9B<<IPPLID<<<<<<<<<<<<GHanuwz���znaUNEDFEGMUanuz���������znaWM��������������������������������������
!!
����������������������������;HTalhfba\TRH;82015;)*2666)&
')+)	





��� ���������)FNU[ZN9���������������������������������������������#0AJVXX[UI?60*)0<ACIKJI?<90.+)))))����������������������������������������Y]bgst��������tkg[Yt�������������vsrnqt���������������}����]benpqvw{|{nkcb_]]]]�+6;A=)�������EOU[hpohf[OLEEEEEEEEZ^gt����������zpg`\Z��������������������<AHNU`aba`acba]UHA<<lnyz|��ztnfjllllllll����������������������
������������K�>�(������4�A�M�X�l�x�|�y�s�f�Z�K������~����������������������������������������������������$�(�0�5�1�0�$�������������������������������������������#���	�����
�#�<�b�nņňŅ�v�b�U�<�0�#�G�C�;�5�7�;�>�G�T�^�[�T�G�G�G�G�G�G�G�G�����������������������������������������
�	�	��
���#�+�/�6�;�/�(�#��
�
�
�
����߾׾Ծվ׾��	��.�;�A�B�;�.����.�"��׾������������ʾ�����"�.�:�E�B�.��ƠƓƉƇ�w�uƁƚƳ������	�
��������������������������������������������������Y�M�Q�Y�e�i�p�r�{�~�������������~�r�e�Y�<�9�/�(�/�<�H�P�U�X�U�H�<�<�<�<�<�<�<�<�Z�N�Z�[�f�s�|�t�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�Ŀ����������ĿĿѿֿݿ�߿ݿӿѿĿĿĿ��0�&�(�0�5�=�I�V�b�i�k�b�X�V�I�=�0�0�0�0�������,�3�Y�r������{���l�Y�4�'���E�E�EuEiEhEiEmEuE�E�E�E�E�E�E�E�E�E�E�E��ʼü����������ʼּؼ�����ּʼʼʼ�¦ ¦¦°²·¿������������¿¾²¦�����k�b�d�r�������������"�:�6����������������q�k�a�a�g�s�����������������������z�x�r�w�zÇÇÓàåìíîìäàÓÇ�z�z�"����/�:�;�D�N�T�W�a�d�a�^�T�H�;�/�"��������������!�&�$�!���	������	����������	���"�/�/�4�/�"��	�	�	�	�x�q�o�l�g�j�l�v�x�z�������������������x���	�������	�����!�!������������������������,�.�*�'�!�����~�t�����w�{���������������������������~�����������������������������������������������y�l�c�b�m�y�����������ǿѿֿĿ�����������������"�������������~�e�^�s�����������ɺֺ�����ֺ������~�����������������������������
������ֹܹԹֹܹ�����ܹܹܹܹܹܹܹܹܹܹϹ̹ù������ùȹϹֹֹҹϹϹϹϹϹϹϹ������������������������������������������������� �)�,�-�)���������_�]�Z�]�_�c�l�p�s�l�_�_�_�_�_�_�_�_�_�_�!��!�-�:�<�F�S�]�S�Q�F�:�-�!�!�!�!�!�!���׾޾������	��������	�����[�P�B�6�4�2�5�:�B�O�[�hāċĘčą�t�h�[�.�$�	����׾��������ʾ���	�� �"�0�6�.����������������������������������������������������$�$�$�����������������������u�i�f�x�������ܺ�'�'������Ϲ������������������������������������������=�#����$�0�@�9�=�V�X�b�o�{�{�v�b�V�=�-�*�$�-�:�F�O�F�<�:�-�-�-�-�-�-�-�-�-�-���������(�-�5�N�Z�Y�]�_�\�N�5�(�����ݿѿǿ����������Ŀѿݿ�����������ݺ@�3�$��*�3�9�@�L�Y�e�r�~�����~�r�Y�L�@���y�s�i�g�k�s�������Ŀʿǿǿǿÿ��������ѿпοѿӿֿݿ�������
�������ݿѿ�������������������������� ���������������������(�+�5�@�A�H�E�A�5�(���ìéììù����������þùìììììììì�[�Q�O�L�O�[�^�h�k�t�u�t�l�h�[�[�[�[�[�[ŠřŔőőŔŔŠũŭŹſ��ŹůŭŠŠŠŠùïàßÙÇ�|Çàìö���(�6�-������ù����������������"�)�*�)������������������������������������������������v�v���������нݽ��ݽݽ�ݽϽ���������������
��� �(�*�(� ������������ĿĿ������������������ĿͿѿ׿ؿѿſĿĽ.�)�,�.�:�G�I�M�G�:�.�.�.�.�.�.�.�.�.�.�U�H�<�3�#� ����#�/�H�U�`�b�\�a�f�[�U�s�t¦²����������������¿¦ŔœŇń�{�u�{ŇŔŠŭŸŶŭŭŠŔŔŔŔ�ܻѻлͻлܻ�������
������ܻܻܻܽr�h�n���������Ľнֽ��ݽнĽ��������r�Ľ����������ĽϽнӽѽнĽĽĽĽĽĽĽ�čā�{�{āčĚĦĿ������������ĿķĦĚč����ĹļĴĸ���������/�-�#��
�������̿	��������ܾ������	������	�	���������������ǾʾѾʾ������������������������������ʼּּܼ߼ּռʼǼ�����������������
�� ������������� I C 4 W ' \ f + : I \ f J ^ T 8 J N Y : [ , D # ; 8 R T x > a ; J - T O a c x J b � \ (  n _ L U < G K < $ # - ? 6 f & . Y f . : F S A 6 q @ o = Z Y $ R Y f S  �  �  �  t  �  F  �  �  �    �  �  W  8  \  �    -  �  �  �  S    8  �  )  �  !  *  W      )  �  =  j    �    P  #  �  k  �  �  Y  �  t  �  ,  #  \    N  �  (  y  �  y  n  �  �  �    �  �    >  �  �  �  �  �  �      �  I  �  �D��;�o�D��;�o�e`B�D���t��e`B���ͽ�o��7L�e`B�C���9X��o��1��`B��+��h��`B����]/�Y��+������h�H�9����h�\)�C��'�`B�q���@��\)�t��o�����\)�Y������aG���w�49X�hs�'��w�8Q�Y���%��o��xս�+�e`B�����]/�Y��y�#���m�h��%�\�}󶽁%�}󶽣�
���
��7L��O߽������w��l���xս����vɽ�"ѽ��`B>�B$�,B��B!d�B 2eB!��B'�B�BB��B1mA���B#�4B��B!Br�Bj�B��B 1TB�B i BoBBB#hB�+B�B+�~A�NKB @B!�B��BXZB	�B*��B)��B�B�BfB�VA�6�B��B!0�BpnB�OB\*BL�B¸B	�B�6B&�(B	f�B&�B6�BdnB"��B��B�BxA��xBrB$YBzhB�BN B�B%��B&�BH<B
�B
NB
ɠB	,B(>�Bb(B/^B	�7B7 B��B�B,j-B-�dB�LB$?�B�gB!Q�B ?B!��BA�B2BG�B0�A��"B#�6BB�B ǜBX(BWVB��B��B��B M�B	�B�KB�KB�8BJ�B+�A�eLB˧B!�uB�B~�B8B*�1B)��B��B�B8�B;^A�pB"bB!B@�B	�B8�B?�B
�B<BB�B&GnB��B&��B�oBàB"CB�?B��B��A���B��B$B\�B�=B@0B�B%JB%�PBF�B��B
��B
�`B<3B'��B=�BhB	�B��B��B=�B,;�B.5iA;yAIU�B	AIo8A��AeVVA� .A��AZ�AWN;B<�@�b?�ɩA��A@�AzmB%�@� CC��@���A�e�A�=�A�#�A�S�A��	@Y�A���@�x�A���A�P�A�-@A�0^Ar3�A��@"3TA�}�>��u>�*�A��.AՔ�@��'@|�AX�tAڞATN�A�VzB�*=���A�	)B[�@z��A���A{�?�;�Ar��A~�A��A��|A͜�A���A���A�INA�rIA���A"v�A2gDAw��A~A�}A�TEA���@���A"zSA&|�Aߩ�A�H�AZVAN��@�KtA�5A8k/AJ�/B	.�AH�>A�Ae#�A��A���A[�AWmB�@�?�:�Aā�A@�ZAy|B�@��C��@�AA���A���A��A�w(A�U@^8|A�_c@�`2A��~A�z A���A��PAo�A�_h@#ɴA�.>�y>F`xA��/AՄ�@�;m@��
AY�A�xASR�Aр_B��=��LA�I�B�,@w�wA���Az�,?�WbAr��A A�E4A���A�hA��A�A�?�A�_OA�|#A!)A1D�Ax�lA	�A�Z!A��+A�3T@�h=A"�A&��A�[A���AYl�AM=-@�zZA�               :               8   :               	      E            X   $   "         	      
                  $      
                     /            i      (               E      	               N      
   )                                                !            '            !   9   -                     1            ?   )                                    '                                       1      '      !                              3         #               !         !         !               !                        !   1   !                                 7   )                                    '                                       #                                          %                        !         !         !            O�+�N���O}�N;�sO�S�N/�NT��N���O���P6�O���Nd�O N'�N>��N��XN�IjO���N�O�NB�4OP��P���P��N�r�O%�,O �N���OM@N���Oj@O��N��_O�v�N��"O�L�N��2N-��N@�(N�L�N3
OM���N$.�O�OmbN���N
-aNfAO��N�O��N %9O��rOL|�O! �O�µOA�O.��O8��N$JN`�N�:�O��N��TO L1OS{IN���N�3�N��O*��O���N���NtZ�O�HNgXyO�O�i�N���N2��N�O�M�]�  �  �  �  �    X  �  �  �  �  �  n    �  5    �  f  �  �    �    �  �  \  �  �  �  �  �  {  !  )  �  N  +  .  K  �  #  �  ^  �  �  �  �  �  l  c    �  S  �  	�  t  �  �  d  �  )  	�  9  �  A    }  �  �  u  �  �  �  K  �  �  �  �  -  ]<D��<t�;�o;�`B��1��o�D���o�o��o��/�#�
�49X�49X�D���T���e`B�,1�u��1��C���w����ě���9X���
��j��j��1��9X���ͼ�j��j��j�ě��+����������`B��`B��`B���#�
�<j�0 Žt���㽉7L���0 Ž�w�,1�49X�L�ͽD���<j�D���P�`�H�9�L�ͽP�`��O߽]/�]/��o�aG��ixսu��%�y�#�y�#��+���P��t���^5��E���Q콶E��Ƨ�Ƨ������
"($
�������

����������������������������������������achmz����������zmfaa���������������������������������������./<HOUXYURH<2/......(6BW[hjssjeXOB6/*!#6Oh����~th\O*
HKNT[amz������zmaTNH���

���������EO[hqt���thb[XOLHEE����������������������
�������������IOR[dhtuwtoh[YOMIIII��

��������������������������������������������������������������������")10)��������)B[rwxTRIB5)��������������������������������������)6;BJUNB96-)����������������������			 ����������FO[ht��������th[RMGF������������������������������������������������������������9BEO[hijiihg[ROLCB99��������������������������������������������������������������������������������)6;76)?BHO[hjhd[OB????????//1;>=>BF;7/**+.////^agnzz�zqna]^^^^^^^^����������������������������������������EN[gty}tkg`[NDCEEEE������������������������������������������������������������������������GUanz������znaUHA@CG5<AHIU\ZZUNI<6555555?EN[t�������{g[NH?<?<<IPPLID<<<<<<<<<<<<FHUalqtz���znaUOFEGFZanyz����������za^UZ�������������������������������������������
!!
����������������������������6;=HTacedaZTOH;:3126)*2666)&
')+)	





��� ����������)5BB>/)���������������������������������������������#0<CILLH<0-#)0<ACIKJI?<90.+)))))����������������������������������������dgltu����������tg[_dt�������������vsrnqt���������������}����]benpqvw{|{nkcb_]]]]�*68=6)�������EOU[hpohf[OLEEEEEEEEagt}��������tggb_\aa��������������������BHPU]abaa[UHB=BBBBBBlnyz|��ztnfjllllllll����������������������
������������K�>�(������4�A�M�X�l�x�|�y�s�f�Z�K������~���������������������������������������������������$�.�0�3�0�.�$������������������������������������������I�<�0�*�!��#�0�<�I�U�b�n�s�w�q�j�b�U�I�G�C�;�5�7�;�>�G�T�^�[�T�G�G�G�G�G�G�G�G�����������������������������������������
�	�	��
���#�+�/�6�;�/�(�#��
�
�
�
����߾׾Ծվ׾��	��.�;�A�B�;�.����	��׾����������ʾ����%�.�4�>�;�.�"�	������ƯƧơƚƙơƧƳ����������� �����弤���������������������������������������Y�M�Q�Y�e�i�p�r�{�~�������������~�r�e�Y�<�9�/�(�/�<�H�P�U�X�U�H�<�<�<�<�<�<�<�<�Z�N�Z�[�f�s�|�t�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�Ŀ����������ĿĿѿֿݿ�߿ݿӿѿĿĿĿ��0�&�(�0�5�=�I�V�b�i�k�b�X�V�I�=�0�0�0�0��	���'�@�M�Y�f�l�o�i�^�Y�M�@�4�'��E�E�EuEiEhEiEmEuE�E�E�E�E�E�E�E�E�E�E�E��ʼɼ��������ʼּּܼܼ̼ʼʼʼʼʼʼʼ�¦ ¦¦°²·¿������������¿¾²¦�������~�t�p�|������������#�&� ��������������q�k�a�a�g�s����������������������Ç�z�z�u�z�{ÇÏÓßàêìíìâàÓÇÇ�"��#�&�/�6�;�H�T�\�a�b�a�`�Y�T�H�;�/�"��������������!�&�$�!���	������	��������	��"�+�/�0�/�"��	�	�	�	�	�	�x�s�q�q�m�i�l�l�x���������������������x���	�������	�����!�!������������������������,�.�*�'�!���������������z�~�������������������������������������������������������������������������y�l�c�b�m�y�����������ǿѿֿĿ�����������������"�������������~�e�^�s�����������ɺֺ�����ֺ������~������������������������������������ֹܹԹֹܹ�����ܹܹܹܹܹܹܹܹܹܹϹ̹ù������ùȹϹֹֹҹϹϹϹϹϹϹϹ������������������������������������������������� �)�,�-�)���������_�]�Z�]�_�c�l�p�s�l�_�_�_�_�_�_�_�_�_�_�!��!�-�:�<�F�S�]�S�Q�F�:�-�!�!�!�!�!�!���������	�������	��������O�C�B�<�8�=�B�O�[�h�t�zĀ��{�t�q�h�[�O�׾Ծʾ¾ʾϾ׾��������׾׾׾׾׾�����������������������������������������������������$�$�$���������������������x�x���������ùܹ��������ٹù����������������������������������������������I�=�2�'���$�0�=�I�V�b�o�x�y�t�o�b�V�I�-�*�$�-�:�F�O�F�<�:�-�-�-�-�-�-�-�-�-�-���������(�1�5�A�R�V�[�]�Y�N�5�(���ѿ˿ſ��������Ŀǿѿݿ߿���������ݿѺ@�8�4�@�C�L�Y�e�r�|�~��~�w�r�i�e�Y�L�@�������y�t�j�h�l�t�������Ŀȿƿƿ��������ѿпοѿӿֿݿ�������
�������ݿѿ�������������������������� ����������������������(�5�=�A�F�C�A�5�(��ìéììù����������þùìììììììì�[�Q�O�L�O�[�^�h�k�t�u�t�l�h�[�[�[�[�[�[ŠřŔőőŔŔŠũŭŹſ��ŹůŭŠŠŠŠ��ùðêçäìù���������$�������������������������"�)�*�)��������������������������������������������������|�{�����������Ľνҽн̽Ľ�����������������
��� �(�*�(� ������������ĿĿ������������������ĿͿѿ׿ؿѿſĿĽ.�)�,�.�:�G�I�M�G�:�.�.�.�.�.�.�.�.�.�.�<�6�/�#�#��#�'�/�<�F�I�U�^�^�Y�`�U�H�<�s�t¦²����������������¿¦ŔœŇń�{�u�{ŇŔŠŭŸŶŭŭŠŔŔŔŔ�ܻѻлͻлܻ�������
������ܻܻܻܽx�k�p���������ĽϽԽݽݽ׽нĽ��������x�Ľ����������ĽϽнӽѽнĽĽĽĽĽĽĽ�ā�}�|āĄčĔĚĦĬĳľľĳĳĦĚčāā������ĿĸĸĻ���������,�*�#��������̾������������	�����	���������������������������ǾʾѾʾ������������������������������ʼּּܼ߼ּռʼǼ�����������������
�� ������������� I C  W  \ f + : F 8 f J ^ T 8 J 7 Y @ [ % D  7 8 H M x > 7 ; J - T Y a c x J b � E  D n _ 6 U + G K B * ! - ? / f & . 5 f .  F S A . q @ o : Z * ' 0 Y f S  �  �  @  t  1  F  �  �  �  F  �  �  W  8  \  �      �  g  �        i  )  �  �  *  W  A    )  �  =  �    �    P  #  �  ;  �  �  Y  �  �  �  �  #    �  T  �  (  y  �  y  n  �  O  �    �  �    >  m  �  �  �  �  �  I  �  �  I  �    F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �  �  �  �  �  �  �  �  �  �  k  S  Y  �  ~  l  `  ^  m  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  6    �  �  �  �  s  b  M  5    �  �  �  R  
  �  >  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        S  �  �  �  �  �      �  �  �  5  �  F  �    Y  n    X  W  V  T  S  Q  P  O  M  L  E  8  +         �   �   �   �  �  �  �  �  �  �  �    r  e  S  <  &  
  �  �  �  �  �  v  �  �  �  �  �  �  �  y  k  ]  L  ;  )    �  �  �  �  �  �  �  �  �  �  �  �  �  �    n  Z  D  -    �  �  �  N     �  R  �  �  �  �  �  �  �  s  p  l  Q    �  �  T     �  �   �  .  <  3  +  J  ~  �  �  �  x  P     �  �  =  �  +  �  �  �  n  d  Y  O  D  :  2  )                �      .  A  T       �  �  �  �  c  K  =  $     �  �  T    �  ?  �  3  �  �  �  �  �  �  �  �  �  �  �  ^  *  �  �  }  A    �  �  =  5  1  ,  '  #            �  �  �  �  �  �  	  !  9  Q      �  �  �  �  �  �  �  �  �  o  ]  L  ;  *      �  �  �  �  q  Q  /    �  �  �  �  u  >  �  �  r  .  �  �  �  �  :  �  �  K  �    M  a  e  Y  2  �  �    9  p  �  �  �  �  �  �  �  �  z  R  '  �  �  �  f  1  �  �  �  S    �  }  �  �  �  �  �  �  �  �  �  �  �  �  s  M  #  �  �  �  �  �  }    
  �  �  �  �  �  �  t  T  3    �  �  �  �  \  !  �  �  ?  s  �  �  �  �  �  z  =  �  �  Q  �  �    |  �  �    J    �  �  �  �  �  �  �  �  �  �  �  �  l  ?    �  C  �  �  �  �  �  �  �  �  t  H    �  �  `    �    s  �    ;  b    �  �  �  �  �  {  q  p  m  \  H  1    �  �  �  m  Z  Q  \  L  <  -  #        �  �  �  �  �  �  �  x  c  K  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  J  *  }  �  �  �  �  �  r  f  T  ?  !  �  �  x  ;     �  z  o  �  �  �  �  �  s  d  V  G  9  *        �  �  �  �  �  �  �  �  �  �  x  s  o  g  \  Q  F  <  1  "      �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  x  e  P  ;  $    �  �  �  �  {  z  x  s  h  X  >  "     �  �  �  l  <    �  �  �  s  k  !      �  �  �  �  �  �  �  �  �  �  p  D    �  �   �   f  )  $          �  �  �  �  �  �  �  �  l  U  9     �   �  �  ~  h  R  <        �  �  o  4  �  �  J  �  c  �  Y  �  6  �  �  �  �    !  F  N  ?    �  r  1  �  �  x  C  1  .    +  #      
    �  �  �  �  �  �  �  �  �  �  �  t  O  )  .  (  #        
    �  �  �  �  �  �  o    �  �  n  2  K  C  ;  4  ,  #      �  �  �  �  �  �  �  �  �  �  �  x  �  �  �  �  �  �  �  �  �  �  �  �  t  c  Q  ?  .    
   �  #                                         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      *  F  R  Q  K  J  X  ^  T  D  4  !  
  �  �  ]    �  �  >  �  )  Z  �  �  �  �  �  �  �  B  �  �  l  7    �  =  �  �  }  �  �  �  �  �  6  r  �  �  �  `     �  �  2  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  k  Z  F  2    q    T  }  �  �  �  �  L  �  �    
y  	�  	  S  v  M  �  �  l  h  d  _  [  W  S  O  J  F  A  =  8  9  F  S  `  l  y  �  �  R  c  X  @  "    �  �  r  F  %    �  �  b  �  �    �      �  �  �  �  �  �  v  ^  G  /       �  �  �  �  �    �  �  �  �  �  �  �  �  �  u  l  f  V  E  1      �  �  �  A  K  Q  S  Q  K  >  *    �  �  �  �  �  o    �  ,  �   �  ~  �  �  �  �  �  �  �  �  �  �  �  �  y  H  
  �  ]  +  #  	�  	�  	�  	�  	�  	�  	�  	p  	J  	  �  �  N  �  �    ]  w  )  7  t  q  `  E  %    �  �  �  X  %  �  �  v  -  �  �  �  >  �  �  x  `  H  /      �  �  �  �  �  �  �  r  V  7  �  �  8  �  �  �  �  �  �  �  �  l  R  2  
  �  �  8  �  Z  �  i  �  d  [  S  J  A  :  2  +  $        
    �  �  �  �  �  �  �  �  �  �  �  {  k  Z  J  :  *    	  �  �  �  �  �  �  �  )  '  $           �  �  �  �  �  \  5    �  �  Z  W  [  	7  	u  	�  	�  	�  	�  	�  	�  	{  	C  �  �  z    �  �      �  �  9  -  !      �  �  �  �  �  �  �  �  �  ~  Y  ,   �   �   �  �  �  �  �  �  �  }  g  Q  :  #  
  �  �  �  o  @     �   �  �  �  �  +  ?  <  8  4  -  !    �  �  8  �  �  %  �  D  �    �  �  �  �  �  �  �  �  �  y  `  E  )    �  �  �  �  ~  }  h  S  >  )    �  �  �  �  �  �  �  �  x  l  `  S  F  :  �  �  �  �  �  �  �  �  �  ~  z  v  s  o  l  h  e  a  ^  Z  �  �  �  �  �  �  �  �  �  s  O  ,    �  �  k    �  �  �  u  a  M  8      �  �  �  h  1  �  �  �  �  �  �  U    �  �  �  �  s  ]  G  1      �  �  �  �  �  z  a  I  >  2  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  J  !  �  �  �  K    �  v  �  U  K  E  ?  9  .  "      �  �  �  �  �  �  �  o  Y  7    �  �  �  S  �  �  �  w  Y  7    �  �  ?  �  �  Q  �  m  �  L  �  �  �  �  �  �  �  �  �  S  !  �  �  y  A  �  �    y   �  $  h  �  x  e  Q  :       �  �  �  �    V  (  �  �  ~  �  �  �  �  �  �  �  �  }  q  e  M  )    �  �  �  �  e  H  +  -  "      �  �  �  �  �  {  ^  A  #    �  �  u  0  �  �  ]  F  /      �  �  �  �  �  �  �  �  �  g  I  (    �  �