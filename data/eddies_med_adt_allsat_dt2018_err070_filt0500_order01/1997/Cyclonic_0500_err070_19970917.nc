CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�vȴ9X       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�8       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��"�   max       <�C�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F�p��
>     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��fffff    max       @vu\(�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @�@           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��-   max       ;�o       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B5'G       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�;       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ;L�c   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?�O   max       C���       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          w       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       Pm�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊q�   max       ?�͞��%�       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��"�   max       <u       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F�p��
>     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��fffff    max       @vu\(�     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @�L�           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?����s�     �  ]            
            	   	       
         
            o                   $      	         &      
         K            	   w   
            
         K   
                     )         ,   2   E                            #   O�fNH��O4q%Ni<Od�O�z�Opk�N��zO;P#��N�#�N)��NT?sN�/�N�^�O'H�O��P�8O.[�Nbo�M���O��NZ��P�ON��N�CFO>�Ol8oPm�N^�\N��8N��O@��PL�OY܇O�X�N�ߥN��P �oN���N[uO:&<Oxa�N��	O�%N�ëPk��Oc4+N�BO���O[:O��NmBONO&]O�O�,N7��Pg%�O��O���OXLRN3��NɼLN!��Ok��N=5vO:��OfW�O�CFNvP�<�C�<t�<o;�`B;ě�;�o:�o%   �D���D����o�ě��o�#�
�T���T���e`B�e`B��o���㼛�㼬1��1�ě��ě��ě����ͼ��ͼ�����/��/��h��h��h��h��h�+�C��t��t���w��w��w��w�''0 Ž8Q�<j�<j�D���D���L�ͽY��Y��]/�]/�e`B��%��o��o��O߽�O߽�t���t���t���E���Q��"ѽ�"���������������������#)-)%!BHQT[afggfedaZTC><>Bot�������|tooooooooo+6COTWUROKC6*%������

���������y���������������xuy�����������������������������������������5B[exwp[NB)��
#-# #)/79/.#"
��������������������
#%'# 
���������������mt��������������xtmm�
#0460#
�������������������������#<Ubs���tnC<880#����������������������������������������`amnnoona]``````````dp�������������zmjdd��������������������(+<Bgt�������tg[B:.(��


�������������������������������������������������������������������#Un�������qI<0ntu�������tnnnnnnnnn��������������������������������������������
#%(#
�����nt���������������son 	)6?BC=2)�� ��������������������z��������}zxuzzzzzzz������ ���������������������������)67<>6))jmtz|~~{zmihjjjjjjjj��������������������.6O[^hlnmijhc[OK;2/.ltv�����������}ttrll����� "
���������)01<GIJUXXXVUIG@?<0)t��������������tlijtahm����������zmb\W[a>BBN[[dgnpg[NBB=>>>>z�����������������yz[\cgt|��{{tgc\\[[��������������������./;<<?BC><//(*......��������������������z��������������xvvz�������������������������������������������076)����������������	������������)5=BKB�������P[_gt���������gc][VP��������������������1<>HIU]abbaUMHD=<961#.'+*#!����������������������������������������������������
#/<GCA@;8/#	)5BNPXXZZNE;5)%$! ")`gt�����togc````````���������������
��,�<�G�P�H�<�/�#�����俫��������������������������������������ā�u�h�_�[�U�[�h�tāčĚĩĳĶĳĦĚčā�H�A�<�G�H�U�Z�a�b�a�_�U�H�H�H�H�H�H�H�H�	���׾ʾ¾������׾����	������	�M�A�C�L�f�x������������������������f�M�����������(�.�6�7�+�(������齷���������Ľнݽ�����������ݽнĽ����s�g�Z�N�B�J�N�T�T�Z�]�g�n�s�~�������v�s�5�,�(��
���A�Z�g�}�}�����|�g�Q�N�B�5����������������������
����
�
�� �������������������������������������������ݽӽн˽νнݽ�����ݽݽݽݽݽݽݽ������������������������������������������������������������������������������'�@�N�R�M�@�=�4�'���Y�@�7�M�f�r����������������������r�f�Y���v�u�z�����н��A�M�f�m�Z�4�
��Ľ������������(�4�A�A�M�P�M�C�A�4�(�ƧƤƦƧƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�����������������Ǻ����������������������������������������$�0�1�2�0�-�&�$���b�b�]�b�f�n�y�{ł��{�{�q�n�b�b�b�b�b�b��㾾�����̾̾׾���	��$�)�)�!���	���z�y�m�g�m�z�������������z�z�z�z�z�z�z�z�G�D�>�>�G�T�`�j�c�`�T�M�G�G�G�G�G�G�G�G������������$�)�6�B�G�C�B�6�)������������������������Ŀ̿ο˿ƿĿ����������s�G�5�(�)�5�N�s����������������������¿²²­²³¿����������¿¿¿¿¿¿¿¿�Z�V�Q�V�Z�g�s�s�~�x�s�g�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������ìàÕÔÔÙàåìùÿ��������������ùì���z�a�;�/�/�8�M�W�m���������������������;�7�/�!����"�)�;�H�T�Z�]�d�a�U�T�H�;�3�-�.�2�9�:�:�L�Y�r�����������~�e�W�@�3���������Ŀѿݿ��ݿԿѿĿ����������������	�������	��"�.�;�<�@�;�.�"�����������x�m�g�i�r�~���������ûͻͻû������F�E�F�P�R�S�_�l�w�x�x���x�l�_�S�F�F�F�FìäàÜàìù��������ùìììììììì���� ���������*�+�/�2�2�5�4�*��������~��������������������������������������������� �	��������	�������ú��ººɺ��!�-�:�N�V�U�:�-�����ֺû-�-�-�4�-�-�-�6�:�F�S�_�`�a�_�S�F�B�:�-��d�T�P�[�t¦��������������	������8�/�,�*�/�7�3�;�H�O�]�a�m�w�x�m�a�T�H�8�/�#�"��"�"�/�5�;�H�J�P�M�H�H�;�/�/�/�/�H�@�;�*�"� ��"�/�;�H�a�m������z�m�T�H�\�O�C�=�>�C�O�\�hƁƎƚƧƺƭƚƁ�u�h�\àÖÓÏÇÆÉÓàèù����������þùìà�[�W�O�N�O�[�h�tāąāā�t�h�[�[�[�[�[�[�������*�3�6�C�D�C�6�*��������������� ����'�@�L�Y�e�j�e�Y�@�3����;�#������0�I�\�n�{ŇőŇ�t�b�U�I�;ŠŘŞŠŭŹ��������ŹŭŠŠŠŠŠŠŠŠ�Q�?�0�=�S�l�����н�����ݽĽ������`�Q����r�f�X�Q�Q�T�Y�f���������������������g�c�j�tčĚĭĳļĿ����������ĿĳĚ�t�gĿĶĳıĬĭİĳĿ��������������������Ŀ�V�P�I�G�=�I�M�V�b�c�n�b�V�V�V�V�V�V�V�V�����������������ùϹѹعϹ˹ù���������E�E�E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�X�\�f�y���������ʼּּܼѼ���������f�X��޼�������������������������������(�*�5�8�@�>�;�6�5�(�������ݿտпп̿ѿݿ������"�"����`�Y�W�_�a�zÇÓàìù������ùìÓ�z�n�`ìåäëìù��������ýùìììììììì P ` d ? 6 = 7 6 G " � @ : 4 k S L T . % s 7 e : � B : s V 7   7 . ; 4 R ; ` 4 J g P Q g Z e < B 5 4 u E S r 9 J ` \ C ^ -  x Z v z ; 6 F I  �  ]  �  �  �  �  �    ,  �  �  X  q  �  �  �  �  �  q  t    ]  �  �  "  �  �  Q    y  �  �  �  �  �  ;  �  �  b  �  �  �  @  #  �  �  �  �     2  !  e  �  �      h  �  U  R  �  X  -  R  �  �  �  �  h  ��e`B;�o�ě��o�49X��1�49X�t��D���\)�T���t��D�����
��t�����0 ž   �C����ͼ�9X�Y���j�u��h�o�#�
�8Q콁%�+��P�\)�,1�����D���m�h���0 ž�-�8Q�8Q�P�`��7L�D����\)�D����]/�q����+�u��7L�y�#�ixս����t��ixս�����`B�+���T��t����T���vɽ��-������hs���mB�B�ZA���B� B0k�B"^B�EB"5�BzB��B��B5'GB0�BĦB%�B$ǠB X�B&��BJB`AB��B ��Bq�B	pQB��B!s�B�jB-
�B':�B�B �`BowB�NB�bB�B!f�B @�B��B�B-A�IpB!�B�;B9�B#	�B&�B�MA��uB}kBsB	��BIgB�(B�KB�^B�Bz�B��B��B6_B	��B;BnB�,B+��B.�B�*B�Bd�B	�B��B�PA���B��B0@�B"?XB !�B"?�B>�B��B@�B4�;B8oB��B�#B$U�B �B'@RB7FB�bBJB Q,Bv�B	�_B@�B!��B��B-eB(A�B�VB ��BA�BxoB�XB��B!ҪB ?�B�VB<B��A�n�B��B?B=rB"OeB&��B��A���BDPB�`B	8>B�OB�YB�/B��B�UB��B< B��B�B	�B
��B�B�gB+�B.�NB��BH�B<GB	��A���Atx�A��A��AVF�AD��A2O&A)�\A��A�0�A��9AK�hA*�eA���@�r�@���@���A+��A7]UB^R@'+�B�=A��AYsA��Af\�AՙAt�YA��A���A�x$A�z�A���A�vCA��H?��AzB�A^�@�1
@���A�`A�}nA��A���@ZR@�/�A��YA���A�E�A�LB�A�5%A۴�A��p?��_A�4�A��A!�@�}A�kQA�{zB�G;L�cC���@�9KA�GA��A�u�A�"A�&KA�{hAu+�A�waA�n�AV�vAC�A1^�A)q�A�u�A���A�xPAL�mA+�A���@���@{@ݖ�A,�XA7Bt�@+�gB��A�?�AY@A��AgA�q3Av�A���A��PA��IA� A�L�A�|�A�?�e.Az�[A]�@���@���A̒�A�~A��A���@S��@|2�A��mA�~A�evA�\ B��À�A�s�A��c?��A��:A���A%��@�A�u;A��B,�>?�OC���@�$A�A��JA��IA���A�x�             
            
   
   !   
                     p            !      %      	         &               L            
   w   
            
         K   
                     *         ,   2   E                            $      !               !            +                     %   =                  )               7               5      %         #                  )      3                        %   #      7      #               #                                             !                        +                                 7               !      #                           )      +                           !      7                                    O��-NH��O4q%Ni<N��Ow�OCN�N��zNF��O�N�#�N)��NT?sN�/�N�^�O'H�OM�6P�rNؐ�Nbo�M���O�©NZ��NƖ8N��N�CFO>�OUNPm�N^�\N��8N��O.F�O���O<jO��MN�ߥN��OI��N�7�N[uO:&<OL�N�f�O�%NHX�P��Oc4+N�BO�*O[:O��NmBONO&]O���O���N7��Pg%�O��O�C�O/DjN3��NɼLN!��OWLN=5vO:��OfW�O�CFNvP�  �  �    �  �  �  �  �  Z  o  �  8  "  ;     q  �  	,  �    x  L  �  T  �  �  !  :  �  �  {  �  2  i  =  �  �  �  �  c  �  �  �  �  �  e  6  �  �  f  E  *  �  �  Z  �  /  )  	_  �  ?      '  �  Q  d  ,  .  �<u<t�<o;�`B%   �o��o%   �ě��t���o�ě��o�#�
�T���T����9X�P�`��1���㼛�㼼j��1�@��ě��ě����ͼ���������/��/��h���e`B���o�+�C���9X��P��w��w�,1�#�
�',1�}�8Q�<j�@��D���D���L�ͽY���%�e`B�]/�e`B��%��\)��+��O߽�O߽�t������t���E���Q��"ѽ�"���������������������#)-)%!BHQT[afggfedaZTC><>Bot�������|tooooooooo(*,6:CKOPOMJC@6*&$%(�������������������y�����������������{y����������������������������������������5BN[fkssogNB)
#-# #)/79/.#"
��������������������
#%'# 
���������������mt��������������xtmm�
#0460#
�������������������������#0<IUbmsvvobUIB60#����������������������������������������`amnnoona]``````````fr��������������zlef��������������������S[[gt����tg[WSSSSSS��


������������������������������������������������������������������#Un�������qI<0ntu�������tnnnnnnnnn�������������������������������������������
#$'#
�������y{��������������}yy)69AB;0)���������������������z��������}zxuzzzzzzz������ �������������������������������� )066;=6)(jmtz|~~{zmihjjjjjjjj��������������������06BO[ehjlljh[OMF=410v�����������~vvvvvvv����� "
���������4<IUVVUUIBA<44444444rutw�������������wtrahm����������zmb\W[a>BBN[[dgnpg[NBB=>>>>{������������������{[\cgt|��{{tgc\\[[��������������������./;<<?BC><//(*......������������������������������������~|}��������������������������������������������076)����������������	��������������&.3)�����W[ggt��������tge_[[W��������������������1<>HIU]abbaUMHD=<961#.'+*#!����������������������������������������������������
#/<GCA@;8/#	)5BNPXXZZNE;5)%$! ")`gt�����togc````````�����������������
��)�/�<�D�L�H�/�
���񿫿�������������������������������������ā�u�h�_�[�U�[�h�tāčĚĩĳĶĳĦĚčā�H�A�<�G�H�U�Z�a�b�a�_�U�H�H�H�H�H�H�H�H��߾׾̾ʾžʾӾ׾�����	���	�����f�Z�M�K�R�_�f�s��������������������s�f���������������(�)�2�3�(����������������Ľнݽ�����������ݽнĽ����g�]�Z�Z�Z�g�s�v�������s�g�g�g�g�g�g�g�g�A�5�(�����(�5�A�N�Z�i�u�}�{�t�g�Z�A����������������������
����
�
�� �������������������������������������������ݽӽн˽νнݽ�����ݽݽݽݽݽݽݽ������������������������������������������������������������������������������'�@�N�R�M�@�=�4�'���Y�X�Q�M�D�N�Y�f�r���������������r�f�Y�������������Ľݽ����6�=�;�(���⽷�����4�1�(��������(�4�5�A�H�H�A�6�4�4ƧƤƦƧƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�����������������Ǻ�����������������������������������������$�/�0�1�/�,�$���b�b�]�b�f�n�y�{ł��{�{�q�n�b�b�b�b�b�b�����������	�����	�������������z�y�m�g�m�z�������������z�z�z�z�z�z�z�z�G�D�>�>�G�T�`�j�c�`�T�M�G�G�G�G�G�G�G�G������������$�)�6�B�G�C�B�6�)������������������������Ŀ˿ͿʿſĿ����������s�G�5�(�)�5�N�s����������������������¿²²­²³¿����������¿¿¿¿¿¿¿¿�Z�V�Q�V�Z�g�s�s�~�x�s�g�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������à×Ö×Ûàçìùþ������������ùìàà�����z�[�N�K�Y�a�m�����������������������;�:�/�$��� �"�/�;�H�T�Y�[�b�a�T�P�H�;�3�1�<�=�<�>�L�Y�r�~���������~�e�Y�R�@�3���������Ŀѿݿ��ݿԿѿĿ����������������	�������	��"�.�;�<�@�;�.�"������������������������������»������������S�Q�S�S�S�R�S�_�l�v�v�x���x�l�_�S�S�S�SìäàÜàìù��������ùìììììììì���� ���������*�+�/�2�2�5�4�*���������������������������������������������������	��������	�������������ú��ººɺ��!�-�:�N�V�U�:�-�����ֺû:�6�.�8�:�F�S�T�^�S�F�?�:�:�:�:�:�:�:�:¿²¦�i�a�m�t¦¿������������¿�8�/�,�*�/�7�3�;�H�O�]�a�m�w�x�m�a�T�H�8�/�#�"��"�"�/�5�;�H�J�P�M�H�H�;�/�/�/�/�T�H�;�,�"� ��"�/�;�H�a�m�|���z�m�a�Z�T�\�O�C�=�>�C�O�\�hƁƎƚƧƺƭƚƁ�u�h�\àÖÓÏÇÆÉÓàèù����������þùìà�[�W�O�N�O�[�h�tāąāā�t�h�[�[�[�[�[�[�������*�3�6�C�D�C�6�*������������(�:�@�L�S�Y�_�c�c�Y�L�@�3�'��I�?�+�#��
�
��#�<�I�U�b�n�{ń�o�b�U�IŠŘŞŠŭŹ��������ŹŭŠŠŠŠŠŠŠŠ�Q�?�0�=�S�l�����н�����ݽĽ������`�Q����r�f�X�Q�Q�T�Y�f��������������������ā�t�h�f�l�uāčĚĦĳľ��������ĿĳĚāĿļĳĳĭİĲĳ����������������������Ŀ�V�P�I�G�=�I�M�V�b�c�n�b�V�V�V�V�V�V�V�V�����������������ùϹѹعϹ˹ù���������E�E�E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�f�Z�]�g�z���������ʼмʼƼ���������r�f��޼�������������������������������(�*�5�8�@�>�;�6�5�(�������ݿտпп̿ѿݿ������"�"����`�Y�W�_�a�zÇÓàìù������ùìÓ�z�n�`ìåäëìù��������ýùìììììììì P ` d ? : 1 4 6 ?  � @ : 4 k S . Y + % s 0 e = � B : l V 7   7 2 5 1 O ; ` * @ g P 9 Y Z V ; B 5 5 u E S r . E ` \ C ?    x Z l z ; 6 F I  �  ]  �  �    �  �    X    �  X  q  �  �  �  �  �  �  t      �  �  "  �  �      y  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       !  e  �  �  ;  �  h  �  U  �  w  X  -  R  )  �  �  �  h  �  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �  �  �  �  �  x  X  5    �  �  �  e    �  a  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  z  x  x  w  w  l  _  R  E      	    �  �  �  �  �  �  �  l  :  �  �  I  �  �     �  �  �  �  �  �  n  [  F  1      �  �  �  h  5    �  �  a  �  �  �  �  �  �  �  �  �  �  �  �  x  W  2  	  �  �  ,   �  S  k  �  �  �  �  �  �  �  �  r  ^  B    �  �      \   �  �  �  �  �  �  �  �  �  �  �  q  V  4    �  �  �  �  A   �  �  �  �  �  �  �  x  j  [  I  5    �  �  �  k  0   �   �   h  9  3  -  &    3  P  \  `  ]  U  F  0    �  �  �  Z     �    A  a  m  m  b  T  C  /    �  �  �  �  �  K    �  �   �  �    I  I  A  5  &      �  �  �  �  �  n  U  :       �  8  4  1  -  )  &  "         �   �   �   �   �   �   �   �   �   }  "        �  �  �  �  �  �  �  �  �  �  �  n  U  <  #  
  ;  3  *      �  �  �  �  �  �  t  U  3    �  �  r  =       �  �  �  �  �  �  �  �  �  �  �  x  e  R  @  3  '      q  l  g  b  Z  S  K  =  -            �  �  �    6  a  �  �    (  -  �  �  z  j  W  C  -    �  �  �  z  D  �      �  =  �  �  	  	+  	%  	  �  {    �  T    �    ~  l  �  �  �  �  �  �  �  �  �  �  �  �  m  B    �  �  �  W    �      �  �  �  �  �  �  �  �  �  �  �  �  �  ~  p  S  6    x  x  x  y  y  y  z  {  |  ~    �  �  z  ^  A  %    �  �  3  I  I  ?  0      �  �  �  �  f  3  �  �  I  �  `  �    �  �  �  ~  {  x  u  r  o  l  f  \  R  H  >  4  *  !      *  x  �  �  �  �  �  �       1  J  T  J  4    �  A  �  u  �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  d  Y  M  B  7  �  �  �  �  �  �  �  �  y  m  `  R  D  5  %         �  �  !    
     �  �  �  �  �  �  �  �  �  p  Y  B    �  �  �     9  4  -  *  &         �  �  �  �  g  5    �  �  �    �  �  �  �  �  j  N  1    �  �  �  n  B    �  z    �    �  �  �  �  �  �  �  �  �  �  u  ]  E  +     �   �   �   �   v  {  \  >    �  �  �  �  �  h  @    �  �  �  X    �  �  �  �  �  �  �  �  �    k  U  ?  *      �  �  �  �  �  }  `  1  2  1  .  ,  *  '  "    	  �  �  �  �  x  Z  <    �  �  #  �  �    0  K  b  i  \  J  :    �  �  8  �  =  �  �  �  ,  ;  =  :  /  !    �  �  �  �  y  Q  &  �  �  z  -  �  �  y  �  �  �  �  h  6  )  '    �  �  �  ~  G    �  T  �  �  �  �  �  �  �  �  �  �  �  �  �  u  X  8     �   �   �   �   �  �  �  �  �  m  W  @  *    �  �  �  �  �  }  ]  !  �  �  d  
�  �  \  �  4  s  �  �  �  �  �  t    �  	  5  
  �  0    ^  `  c  X  K  <  -         �  �  �  �  �  �    Z  i  x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  5    �  �  t  e  T  A  +    �  �  �  j  V  C  /  $    
  �  �  T  �  �  �  �  l  R  7    �  �  �  W     �  �  Z       �  p  �  �  �  ~  o  _  M  :  #    �  �  �  �  m  P  0  �  �  �  �  �  o  \  G  (  �  �  �  �  �  �  l  F    �  �  �  _  b  c  d  b  Y  P  E  6  (      �  �  �  �  �  �  �  �  u  �  �  �    '  6  $  �  �  �  k  )  �  V  �    �  r    �  �  m  [  O  E  7  (      �  �  �  �  �  �  �  �    \  :  �  �  �  �  �  �  �  x  a  I  -    �  �    <  �  �  	  c  c  f  a  ]  Y  V  T  T  N  >  '    �  �  �  �  w  R  �  C  E  #    �  �  �  �  m  N  R  L  :  !  	  �  �  �  �  �  5  *      �  �  �  �  }  [  ,  �  �  �    L    �  i  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    n  Y  �  �  �  �  �  �  �  �  x  j  [  K  ;  +    �  �  �  �  w    $  2  E  U  Z  T  H  5      �  �  �  B  �  �  C  �  /  �  �  �  �  �  �  �  �  �  r  P  (  �  �  �  �  N    �  �  /  ,  *  '  $  "             �  �  �  �  �  �  �  �  �  )    �  �  �  N  #     �  �  �  |  �  �  �  Z  .  �  P  S  	_  	K  	$  �  �  �  e  W  +  �  �  ?  �  {  C     �  �  /  �  �  �  �  �  �  k  D    �  �  .  
�  
J  	�  �    6  :  �  �  $  $  9  >  <  6  .  #    �  �  �  �  ^  &  �  �  8  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  Q    �  �  �  �  n  5  �  �  e    �  �  �  )  '    �  �  {  &  �  {  F  B  $  �  �  ~  E    �  �  [    �  �  �  �  �  �  �  l  F    �  �  �  `  5  �  �  z    :  Q  C  5  '        �  �        (  =  S  h  i  e  `  \  d    �  �  �  {  L    �  �  e    �  o    �  i    �  w  ,       �  �  �  �  `  +  �  �  j  '  �  �  u  U  3  �  �  .    �  �  �  `  )  �  �  `    �  �  S    �  <  �  U    �  �  �  u  d  I    �  �  �  H    �  �  [  "  �  �  y  B