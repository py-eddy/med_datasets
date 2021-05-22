CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�333333        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��a   max       P�}�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       =y�#        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?h�\)   max       @F�G�z�     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v{�z�H     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q            �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�V�            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       =8Q�        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�v4   max       B4�2        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4Ǝ        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�}�   max       C���        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =���   max       C��        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          K        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��a   max       P�        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�.H�   max       ?��p:�~�        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��E�   max       =u        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��\(��   max       @F�33333     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v{��Q�     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�V�            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C>   max         C>        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?uL�_��   max       ?��p:�~�     �  ^�      
                               K   H   $            1                  B      	                  4   	         
   2            )      	   *      0                              
   $   
            #               
               Nz2�O�O��LOM��O?:�N���N"�N�
%N�}O�CyN��P�}�O�+�P=��N��OBH�N� �P��O�@O�?�Nn�'O��OVz~P�dUO��N�O
�&O̴]O���NHF�NKdcP�Nq�O��BN�	�N��'O���N���M��O8�OxE�Ofc�N,\�O܃�N��P���NG� N��N!�P��NqC�P-��N�
�M��"N˗HO.�[OVniN�dOKxMO}�O!sOO�N��OkIN�v�N1��N��
O|DO�UZM��aN�� O-�=y�#<o;�o:�o:�o��o��o�ě��t��t��t��#�
�#�
�D���e`B�e`B�e`B��o��C���C���t����
���
���
��1��j�ě��ě����ͼ��ͼ��ͼ�����/��/��`B��h��h�����o�o�o�+�\)��P��P���������''''0 Ž49X�8Q�<j�@��D���D���Y��e`B�u�����hs������������������9X��9X���������������������������������������� %,5<HV_Z_bcgaU<-" ��$)-444)�����0<BIbmn{zunhbUIE<300��������������������369BOUSOB63333333333��������������������RTWaimppmmkca_VTRQRR��)31521-)����)*568<BCB?95*-)(%fm�������������zl_\f�
#43<HSHAA6/(#���KTmz��������\MFJNMEK����������������������������������������'*36@CFECB<6*#''''�����

����������������thVPO[t��������������)43)���569BHOTOIOQOLB625555GOZ[hjttqkkih_[[OJHG�������������������������'%��������������������������������������������������
#/<>EA</.*#	������������������������������������������������������JOQT[hjljh[OJJJJJJJJ��������������������LNX[gtxtpig[NLLLLLLL#8HUa\]^]YUH<
��������������������STakmz��zmmaZTTRSSSS��������������������xz���������zzwxxxxxx�����������������������))-..)����&)05BN[addc[N50)%&%&agnz����������xnea_az�����������zzzzzzzz������
#&
���������������������������&BPan{������}bI2*(&��������������������������������������������������������� #.<Ibn{���{b<0#NNX[gmkhga[XNMNNNNNN)6BOmx{�����OB841~�����������������}~�������������������� &),/.+)������������������������������������������
#####$#
� ���v������������������vrt������������~ursur��������������������������

����������>BNNN[ag[NB;>>>>>>>>st{�������������~zys�������������#,*$#tt���������vtmjltttt��������������������������������������������������������#/7<?DG<7/'# EHUahnz��znaUTOLJHEE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������ʾվѾӾʾ��������N�A�5�(������A�N�Z�s�u�v�|�s�g�Z�N�$�"�����$�+�0�=�I�V�X�W�T�I�D�=�0�$�����������������ʼּܼ�����ؼʼ����A�>�8�4�A�M�N�Z�_�b�f�o�h�f�Z�M�A�A�A�A��������������������������������������������������������'�.�'�&�����čąā�yāĆčĚĥĦĳĿ��ĿĳıĦĚčč������߿������5�A�Y�Y�N�F�A�(���g�[�Z�W�Z�g�s�|�������������������s�g�g���������������
�nŅŇœŇ�{�n�Q�<�����E(EEEE*E7EAE>ECELE\EuE�E�E�E�EiE]EPE(�� ���������5�N�g�����������s�Z�A���n�m�b�n�zÇÊÓàêàÓÇ�z�n�n�n�n�n�n�<�1�/�#����
��#�/�8�<�H�K�S�T�R�H�<�	����������	��!�"�.�1�2�.�"��	�	�	�	������������������������������������������������
�
����������¶²°¸������àÓÇ�z�n�\�U�Q�Q�U�a�i�n�zÏÚÛââà�F�>�:�2�-�(�-�3�:�@�F�L�S�]�^�S�F�F�F�F�Ϲù��������¹ĹϹܹ�����������ܹϿѿɿĿ����������������Ŀ˿ѿݿ����ݿѼf�`�n�����̼���1�<�-� ��ּʼ������f������������6�C�O�h�o�u�c�C�6�*����Ľ����������������Ľнѽݽ�ݽݽԽнĽ�����'�(�,�4�6�B�[�i�i�h�[�R�O�B�6�)��r�n�X�L�E�H�L�e�r�~�����������������~�r���׾ʾľþ׾����.�;�G�Q�G�;�.��	��ǈ�~�{�o�{�ǈǎǔǚǔǔǈǈǈǈǈǈǈǈ���}�y�m�j�g�m�y�������������������������\�\�L�H�W�o���ɺ����-�]�S�F���ɺ~�\����������������������������������������������׾Ǿ��ʾ׾���	�������	���U�I�M�U�U�a�c�n�r�u�v�r�n�a�U�U�U�U�U�U�"���"�%�*�/�;�=�G�H�L�H�;�;�/�"�"�"�"�/�"�������"�H�T�a�i�k�g�a�S�H�;�/�� ������)�4�-�)��������������������������������������������������[�U�O�C�B�?�C�M�O�[�tāąā��y�t�k�h�[������������������(�0�5�6�9�6�)������t�n����������������ľ˾������������<�8�6�<�A�H�U�Y�V�U�H�B�<�<�<�<�<�<�<�<��������������������Ŀݿ����ѿĿ����M�J�A�<�;�=�A�F�M�Z�f�g�q�f�c�Z�M�M�M�M�����x�l�W�Q�Z�l���û��4�M�N�;����л�����������������������������M�C�@�4�3�(�4�@�M�Y�f�f�m�j�f�Y�M�M�M�M��������������&������������������s�j�]�Z�^�h�s���������������������������������
������
���������������y�}�������������������	�)�#����������������v�r�f�^�_�f�r����������������������������ʼּ޼ּӼʼ��������������������Z�Y�M�B�M�Z�]�f�s�������������s�f�Z�Z�O�G�C�6�������*�6�C�L�\�h�g�\�W�O�x�_�U�W�S�P�Q�S�]�_�l�x�~�������������xÇÁÁÃÁÇÊÓÝàìõùúøìàÓÏÇ��������~�}�~��������������������������ŔōŊŊŔŘũŭ������������������ŭŠŔ�t�o�h�\�T�[�`�h�tāčĚĦĭįĦĚčā�tD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ÓÏÇÇÇÎÓàêéêàÓÓÓÓÓÓÓÓ�z�t�q¦²»¿������¿²¦��������������������������������������������ĽȽнݽ�ݽнĽ������������������������������
��
�	�������������ػ������������������ûɻлػۻֻлǻû�������������	��2�@�M�^�b�]�S�1������!�!�!�.�.�:�:�G�G�G�G�:�.�.�!�!�!�!�!�!FFFFFFF$F1F5F1F-F$FFFFFFFF�����������������ùϹֹڹڹѹϹù������� 7 5 I 2 ; - N H Z ? n A M _ w 6 ; 8 5 2 a C N J a . w 9 p 2 J b N J O < - f J + 3 D _ ) < G d C n X 8 T { O F T K 7 O 7 R W ?  " � ,   a � Y >  �  K  �  �  �  �  ^  �     �  �    �  �  �  �  �  o  �  T  z  I  �  m  �    �  �  �  H  w  &  �  �  �  �  �  �    �    �  s  �  �  �  z  �  >  �  y  C      �  �  �    �    x  K  �  �  �  v  �  #    h  �  �=8Q캃o��9X��󶼃o��`B��`B�T�������h�T����1���T�@����ͽC����㽃o����w�ě��0 Ž+�� Ž8Q�o�t��T���L�ͽ+��h���-�\)�L�ͽ8Q��w������P�\)�8Q콓t��<j�,1���-�L�ͽ�{�0 ŽT���49X�m�h�<j��t��]/�<j�]/�]/���T�aG��m�h��+��O߽�9X��%������hs���㽬1��E����`���T��G���S�B�_B4�2BDB�B'B�3BI�B�A�v4B�vB�B��B+�A�`�B �qB5.B0/�B��B�VB�B8�B\B*�IB,��BzB �kBYB!��B�B@�B<�B�VB܀B��B!�]A�ݵB��B <OB�B��B�8B�EB
�\B�B"~�B(&[BK7B"�OB#��B' �BޥB�iB��B�#BҹB�B ȌBbB�IB
��B� B��BOxB
�(B��B%)�B
 BA�BgkB��BI_BT@B�	B4ƎB��B�9B&�@B�.B��B��A���B��B>�B�B?.A��GB �yBBqB0=�B��BBcBBDBnBA�B*�3B-C�B�B!?NB@_B!��B��B@B�DB��B��B�+B!��A���B��B AlB�5B2B��B�{B
�zB֙B"@�B'�nBA�B"�YB#��B&éB��B�B>�B�B�B7kB ��B9�BǵB
��B�yBCUB<�B
��B��B%@.B
2�B9pB?�B��B?�B?�C�0KAM{�A�5�B
��@�!�A=�hA�<�?zuqA��A�%�A��A�Y�C��5A���A�hDA�g8A]�A��A�/Aș@��S>��Ay-�@���A�E�A'�Aؽ�?�\cAZ�B��Amz�@4V�AsZ�AYZA�wpA��mA�7�A��A�j�A�A�*bAI�SA�\Aw*�A=�Z@���Bpc@�d�@��aA�E�A���A��B@��@�ݒAB�B d�@�(HA�/A�N�A�q�A���C��A�'�A��AA0��A&�A�[:@�ne@�[A��C���=�}�C�(	AM�A��`B
��@���A=,A���?m�nA��A��A���A詘C��FA���A�|_A�G�A[�A��A��A���@�,�>�,WAyk�A�A�b�A(�OA�~?���AZ��B�Al��@DS�AtOAYh�AƁaA���A��lA�~!A�-A���A�~�AI&�A�~�Ax�A>�@�B��@׆@��UA��uA��A��@� -@�*AB��B ��@��A�G�A��A��A܂)C�A�~tA�t�A1�A#��A��@�4�@�|AEC��=���      
      !                        K   H   $            1                  C      
            	      4   	            3            )      
   +      1                                  %   
            $               
                        !                           7   '   /            %                  ;                        ;                                    #      ?            )      -                                                   %                                             /      %                              ;                        ;                                          =            )      -                                                   #         NV�N�i^Oof�O*7�N�gkN���N"�Ni�|N�}OG(�N��PtˇNO�[O�E�N��OBH�N|�uO�-LO�@Oyh�N(  N��OVz~P�dUO]�TN�N�
O̴]O�%�NHF�NKdcP�NX!hO2x�N�	�N��O�4N���M��O8�O1{&Ofc�N,\�O��N��P~��NG� N��N!�P��NqC�P-��N�
�M��"N˗HO.�[O>�N�dOKxMO}�N�bO�N��OkINZܑN1��N��
O|DO�2iM��aN��O-�      �  d  [  �  �    L  u      	m  �  �  �  �  $  �  �  e  ,  C  �  A  �  E  x  <  �  �  �  �  U  t  t  �  X  �  s  �  m  �  f  �  �    �  �  :    �  x  U  �     y  �  �  �  �  
B  �    �  J  �  �  @  9  �  *=u;��
��o�o��o��o��o��`B�t���o�t����
�]/��9X�e`B�e`B�u��󶼋C����
���㼴9X���
���
��/��j���ͼě��������ͼ��ͼ�����`B�\)��`B���P�`�����o��w�o�+�H�9��P�'��������''''0 Ž49X�@��<j�@��D���Y��Y��e`B�u��+��hs�����������㽡����E���9X����������������������������������������*/9<HPUW\__^UM</'#$*��")+./)����9<IUabbjnibUIA<89999��������������������369BOUSOB63333333333��������������������RTWaimppmmkca_VTRQRR�&),,**))����)*568<BCB?95*-)(%dmz�������������zmcd

#$%#








STaz�������zmTOPTTRS����������������������������������������(*56:CECC>6*#$((((((������	
���������������thVPO[t���������
).0)������56=BFOOOJB6355555555IOP[hspjjhg[ROLJIIII�������������������������'%��������������������������������������������������#&/<<DC@<0/+#!������������������������������������������������������JOQT[hjljh[OJJJJJJJJ��������������������MNZ[gpnhg[NMMMMMMMMM#/@HUYWUNH</'#��������������������TTaimz|zzmia\VTSTTTT��������������������xz���������zzwxxxxxx�����������������������))-..)����55BDNT[]aa_[SNB5,+-5agnz����������xnea_az�����������zzzzzzzz������

���������������������������#&Hbo{�������{nT4-+#��������������������������������������������������������� #.<Ibn{���{b<0#NNX[gmkhga[XNMNNNNNN)6BOmx{�����OB841~�����������������}~�������������������� &),/.+)������������������������������������������
#####$#
� ���v������������������vrt������������~ursur��������������������������

����������>BNNN[ag[NB;>>>>>>>>st{�������������~zys�������������#,*$#tt���������vtmjltttt��������������������������������������������������������#/6<>CD<9/(# EHUahnz��znaUTOLJHEE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������žʾо˾ʾ������������A�8�(�#��� �(�A�N�Z�g�p�s�x�p�g�Z�N�A�0�$������$�.�0�=�I�T�U�U�R�I�@�=�0�����������������ʼּ߼��ּѼʼ��������A�>�8�4�A�M�N�Z�_�b�f�o�h�f�Z�M�A�A�A�A����������������������������������������������������'�,�'��������čąā�yāĆčĚĥĦĳĿ��ĿĳıĦĚčč�������������(�5�5�?�A�7�5�(����g�[�Z�W�Z�g�s�|�������������������s�g�g�������������������#�<�b�{Ł�z�]�F�0�
��E7E7E4E7EBECEPESEXETEPECE7E7E7E7E7E7E7E7������5�N�g�s�������������s�Z�N�5��n�m�b�n�zÇÊÓàêàÓÇ�z�n�n�n�n�n�n�<�1�/�#����
��#�/�8�<�H�K�S�T�R�H�<�	���������	���"�-�-�"��	�	�	�	�	�	��������������������������������������������������
�
����������¶²°¸�������_�U�S�S�U�Y�d�l�n�zÈÒÓÙßÞÓ�z�n�_�F�C�:�5�:�D�F�H�S�[�\�S�F�F�F�F�F�F�F�F�ϹŹù����ùϹܹ����������ܹϹϹϹϿѿɿĿ����������������Ŀ˿ѿݿ����ݿѼf�`�n�����̼���1�<�-� ��ּʼ������f��������������6�C�E�\�Z�P�C�6�*���Ľ����������������Ľнѽݽ�ݽݽԽнĽ��)�(�(�)�-�5�6�B�O�[�g�h�h�[�P�O�B�>�6�)�r�n�X�L�E�H�L�e�r�~�����������������~�r���׾ʾžž׾����-�;�G�O�G�;�.��	��ǈ�~�{�o�{�ǈǎǔǚǔǔǈǈǈǈǈǈǈǈ���}�y�m�j�g�m�y�������������������������\�\�L�H�W�o���ɺ����-�]�S�F���ɺ~�\������������������������������������������������������	���������	���U�I�M�U�U�a�c�n�r�u�v�r�n�a�U�U�U�U�U�U�"�!��"�&�-�/�;�;�E�H�K�H�;�5�/�"�"�"�"�;�0�/�&�)�/�4�;�H�T�]�a�a�a�\�T�M�H�;�;�� ������)�4�-�)��������������������������������������������������[�U�O�C�B�?�C�M�O�[�tāąā��y�t�k�h�[��������������������#�)�,�.�)���������t�n����������������ľ˾������������<�8�6�<�A�H�U�Y�V�U�H�B�<�<�<�<�<�<�<�<���������������������Ŀۿݿ��ݿѿĿ����M�J�A�<�;�=�A�F�M�Z�f�g�q�f�c�Z�M�M�M�M�лû��z�_�W�_�l���û���4�I�N�@���������������������������������M�C�@�4�3�(�4�@�M�Y�f�f�m�j�f�Y�M�M�M�M��������������&������������������s�j�]�Z�^�h�s���������������������������������
������
���������������y�}�������������������	�)�#����������������v�r�f�^�_�f�r����������������������������ʼּ޼ּӼʼ��������������������Z�Y�M�B�M�Z�]�f�s�������������s�f�Z�Z�O�G�C�6�������*�6�C�L�\�h�g�\�W�O�x�l�`�X�S�R�T�_�l�x�|�����������������xÇÁÁÃÁÇÊÓÝàìõùúøìàÓÏÇ��������~�}�~��������������������������ŔōŊŊŔŘũŭ������������������ŭŠŔ�h�`�[�X�[�h�tāčĚěĚęčā�t�h�h�h�hD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ÓÏÇÇÇÎÓàêéêàÓÓÓÓÓÓÓÓ�z�t�q¦²»¿������¿²¦���������������������������������������������ĽȽнݽ�ݽнĽ������������������������������
��
�	�������������ػ������������������ûɻлػۻֻлǻû�������������
��3�@�M�]�b�]�R�/�'����!�!�!�.�.�:�:�G�G�G�G�:�.�.�!�!�!�!�!�!FFFFFFF$F1F4F1F-F$FFFFFFFF�����������������ùϹֹڹڹѹϹù������� :  5 $ ; - N ; Z 3 n ; + l w 6 : - 5 3 : C N J W . U 9 p 2 J b > E O 9 % f J +  D _  < M d C n X 8 T { O F T K 7 O 7 D W ?  % � ,   b � W >  o  �  �  h    �  ^  m     �  �    h  B  �  �  u  (  �  �  3  �  �  m  �    $  �  �  H  w  &  m  �  �  �  "  �    �  p  �  s  �  �  s  z  �  >  �  y  C      �  �  �    �    �  K  �  �  j  v  �  #  �  h  �  �  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  C>  �     �  �  �  �  �  �  �    h  Q  ;  )  u  �  �  p  Y  @  �  �  �    
            �  �  �  �  �  �  f  @     �  y  }  }  w  �  �  w  a  H  ,    �  �  �  f  -  �  �  r  {    Q  d  c  _  Z  V  S  K  F  C  *  �  �  �  ;  �  @  �  �  �    0  E  T  Z  Z  W  R  K  C  8  (    �  �  �  y  L  Y  �  �  �  �  �  �  �  �  �  x  n  e  \  Y  a  i  t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    	        
  �  �  �  �  �  �  �  �  z  b  K    �  �  L  H  D  7  )      �  �  �  w  P  ,  
  �  �  {  J    �    .  <  I  X  `  i  r  f  T  >     �  �  �  g     �  �  &    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  D  �  �  E  �  �  T  �  @  �  /  �  A  r  �  �  Q  �  �  �  	  	  	;  	\  	j  	4  �  r  �  �    &  �  �  �  �  �  �  �  �  �  z  d  D    �  �  n  G    �    �  �  �  �  |  f  T  F  [  b  O  ;  (      �  �  �  �  l  �  �  Z  (  �  �  R    �  q  L  /  �  �  �  F      �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  n  e  \  O  A  3  %              !  $    
  �  �  �  �  �  S    �  =  �  e  �  �  �  �  t  V  ?  1  %      �  �  �  b    �  #   �   V  �  �  �  �  �  �  �  �  q  ;  �  �  o  %  �  �  I    �    F  P  Z  d  Z  L  >  5  /  )  "          �  �  �  �  �    )  (      �  �  �  �  �  �  �  �  s    �  C  �  U  �  C  ;  0     
  �  �  �  �  q  j  [  I  8    �  �     *  h  �  f  >  '    �  �  �  s  P  0    �  �  T  �      V  p  �  �    "  :  ?  4    �  �  �  \  #  �  �  �  }  O  /  �  �  �  �  �  �  x  f  P  8      �  �  �  �  �  v  ^  N  >  "  8  D  A  +    �  �  �  �  o  N  -    �  �  Q  �  �    x  w  s  k  `  J  0    �  �  �  �  Z  )  �  �  �  h  ,  k  ;  ;  2      *  8  #    �  �  �  U  *  	  �  �  4  �    �  �  �  �  t  V  1    �  �  z  @    �  �  F     �   z   4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  =     �  P  �  p  �  �  �  �  S  �  �  a  y  �  �  �  �  �  �  �  y  \  =    �  �  �  �  �  i  M  2    "  '  *  1  >  I  R  S  H  7  !    �  �  �  >  �  {     �  t  M  6  "      �  �  �  �  �  �  |  H    �  Y  �  M  �  i  o  s  m  f  Z  L  ;  )    �  �  �  �  Y  #  �  �  �  U    W  �  �  �  �  �  �  �  �  �  �  �  -  �  ^  �  -  T  z  X  N  C  9  ,        �  �  �  �  f  '  �  �  �  Q    �  �  �  �  �  �  �  �  �  |  q  f  Z  N  B  6  )        �  s  i  \  L  6  !    �  �  �  �  �  p  C    �  �  �  b  <    �  �  �  �  �  �  |  K    �  }  !  �  b    �  �  v  `  m  i  c  Y  L  =  ,      �  �  �  �  x  ]  E  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  j  Y  G  5  "    �  �  '  =  H  R  X  a  e  b  L  #  �  �  r    �  U  �  S  �  d  �    u  i  W  B  +    �  �  �  �  y  R  (  �  �  �  x  L  �  �  �  �  �  �  �  y  U  ;  &  	  �  �  R  �  u  �  �   �              �  �  �  �  �  �  s  W  :    �  �  �  �  �  �  �  �  z  t  j  ^  P  8    �  �  �  ~  J    �  �  d  �  �  �  �  �  �  �  a  0  �  �  �  �  �  y  `  G  -     �  :  -      �  �  �  �  �  �  n  U  8    �  �  �  f     �      	    �  �  �  �  �  �  �  �  v  `  K  5    	  �  �  �  x  K  "  �  �  �  �  �  �  �  �  �  �  \    �  K  �    x  c  N  <  T  Y  S  O  J  G  P  @    �  �  1  �  u    �  U  U  U  V  M     �  �  �  }  \  ;    �  �  �  �  l  H  $  �  �  �  �  �  �  �  �  �  o  Z  B  (    �  �  t  4  �  �       �  �  �  �  �  �  �  u  Z  >    �  �  �  �  X  p  �  a  y  w  t  n  c  V  A  "  �  �  �  ~  d  ;  �  �  �  �   �  �  �  �  �  �  �  �  p  [  K  =  F  k  �  ~  z  u  o  c  W  �  �  �  �  x  m  c  W  K  >  1    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  Y  ;    �  �  ~  U  4  -  �  �  �  �  �  �  �  �  �  �  y  N  !  �  �    7  �  }    
B  
  	�  	�  	�  	M  	  �  �  �  g    �  K  �  7  i  j  P  (  �  �  �  �  �  ~  j  V  B  +    �  �  �  �  z  a  I  0      �  �  �  �  |  ^  A  !  �  �  �  �  �  ^  5  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  `  F  (  
  �  �  �  �  J  K  L  M  O  Y  b  k  m  b  V  K  >  0  "      �  �  �  �  �  �  �    p  _  G  ,    �  �  �  �  w  M    �  C   �  �  �  �  �  �  �  �  j  O  3    �  �  �  �  a    �     �  2  9    �  �  �  U    �  i    �  �  o  "  �  �  �    4  9  (      �  �  �  �  �  �  �  �  �  x  n  c  X  M  B  7  �  �  �  �  �  �  q  Y  7    �  7  �  E  �  4  �  �  X  �  *      �  �  �  �  ^  1    �  �  W  �  �    �  &  �  *