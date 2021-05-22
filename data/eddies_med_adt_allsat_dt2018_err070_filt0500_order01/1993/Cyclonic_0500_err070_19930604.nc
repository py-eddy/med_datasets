CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���"��`     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mۖc   max       P{s     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       <#�
     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @F��\)     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v�fffff     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @�@         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��-   max       ;o     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�O�   max       B4m     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4��     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�#	   max       C��g     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��k   max       C�ߴ     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          z     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mۖc   max       Pj�U     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��i�B��   max       ?Ժ��)_     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��v�   max       <t�     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>s33333   max       @F�ffffg     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v�fffff     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @�          0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?Է�4m��     �  b�                           -            %                
   ,   !                        	   ,                                                y                  
   5            ,   
   #         #         
         
            	               	OCQNW�N>a�O%91N��O�O�xOUY7P{sO��Pj�UO���Ps^�O[O��N�pO�G�N�OlO�@O�_O�K�Nrx�O?��Oow�O�N��eO�a�O%�O���O��O��N��*O�uENo�N��&OSDO�n�O��NV��Nm~�Ne�MۖcN0�xNI�PR,�Ob�Nl9�N��N�,�O2�O�2�PDu'O��N۾VN �AP*�O;"�O��N<i�O��O�ճN9oN"σN�J�O+`wOpˇNwGO��N� O17%N�n�N��O�O8��N5	�N�]<#�
<o<o;ě�;��
%   �o��`B�o�t��#�
�#�
�49X�T���T���u��o��o��o��o��C���C���t���t���t����㼣�
��1��1���ͼ��ͼ��ͼ�`B��`B��`B��`B��h��h��h�����o�o�+�+�\)�\)�t��t��t���P��P��w�#�
�0 Ž0 Ž49X�49X�<j�H�9�H�9�Y��]/�aG��aG��aG��q���y�#�y�#��+��\)��\)��\)���P��{��v���
#&--%#
�������!")/3880/%"!!!!!!!!���������������������������������������������


���������������������������������������������)25CFC5)���9JN[t����������gNB69��������������������%BVbht�������t[@6$%��������������������6Ea����������umTG;16����	 �����������
$&%#�������mnz������znmmmmmmmmm )5N[aflmlh[NB5-+)
#/01//%#

)0<I_bnrnjfaXI<0)$$)[bn{�������~{wnkgaZ[Tamz����}|tmaTOLPTz���������������zzzz�������������������� #'/<Hdkie]LH</.*#  ��������������������jno{�����}{nnijjjjjj������������������TTamz~zzuomgaTMKNOTTLS[ht����|{th[OJGIL���������������������������
����������
"#$#
��������qtz��������������yvq���������������������������������������������� ��������������� ������������������������������dnuz{}|znmidddddddddst���������vutssssss��������������������������������������������������������������������������������������&*(*-)  &)257?>95)#&&&&&&&&����������������������������������������;BNgt{�����tg[NDB<.;��$'(&"������������� "�������������������������������������������������������������� BNgt��e[B5�� �������������.9BMSZg�������tdJA<.nn{������{vnnnnnnnnn|����������������||AEOUX[hhy���|thbRE@A��������������������
!
��������������������xz��������������wusx������
����������������������������������������������������������������������_anz������������znc_����������������

	���������������������������� #'/4@HLNNNIH</'$! FHMU^\UH<?FFFFFFFFFF��������������������ÚàÚÙÝàìù������������������ùìÚ�0�-�#��#�0�<�I�J�I�H�<�0�0�0�0�0�0�0�0�t�i�t¦§¦ �
�����������������
��#�/�9�7�/�,�#��
�4�,�)�(� �(�4�4�5�A�G�K�E�A�4�4�4�4�4�4ƳƬƧƙƐƍƎƔƚƧ������������������Ƴ�H�E�=�<�;�C�A�H�U�X�a�f�n�t�v�n�a�^�U�H�0�*��"�$�,�5�=�A�I�V�b�h�c�V�N�I�=�6�0��ѿ��������������Ŀѿ����$�-�,�%������������$�0�6�7�2�0�(�$������	����ʾ������ʾ��.�T�y�����`�V�;�"�	�a�W�O�Q�[�k����������žǾ���������s�a����˿ſȿѿ�����(�A�������s�C�(��"�	���������������	��"�/�6�;�H�I�;�/�"���������������������������������������������������������������������������������6�O�c�h�f�Q�6����ŹŮŴŹż��������������������������ŹŹ�������������(�4�A�M�]�h�d�M�(�����{�t�u�|�����Ľݽ�����߽нĽ�����ƲƜƋƆƁƚƤƳ������������������Ʋ�������������������ʾ˾Ѿξʾþ����������*�������*�6�C�K�V�]�\�O�C�>�;�6�*�;�4�/�*�'�$�"�/�;�H�T�a�e�n�o�m�a�R�H�;FFFF$F+F1F:F=F?FJFUFVF^F_FVFJF=F1F$F�g�[�Z�O�V�Z�g�s�v���������s�g�g�g�g�g�g��	���������������������"�(�/�3�:�/�)��"�"����"�/�0�;�H�J�T�^�]�T�H�;�/�"�"�����������������������л߻�޻̻û��������������������������	��(�(�"����	���s�g�Q�J�L�Z�g�������������������������s�0�.�&�$��$�0�3�=�>�E�C�A�=�0�0�0�0�0�0ŹŭŠŖŔŘŠŨŭŹ������������������Ź����������������������������������������ŭŪŠŔŇńŃŇŔŠŭűűŭŭŭŭŭŭŭÒÌÇ�z�n�`�U�U�a�n�zÇÓáìúöìàÒ�;�"��	��þ��Ⱦ׾�	��/�C�G�U�X�T�G�;�z�t�g�g�o�o�t�{¢¨¦�g�a�g�q�s�������������s�g�g�g�g�g�g�g�g�������������
�����
����������������Ľý������Ľн׽ҽнĽĽĽĽĽĽĽĽĽļ�����{������������������������������������������ʼҼԼʼ���������������������ǈǀ�{�p�{ǀǈǔǗǚǔǌǈǈǈǈǈǈǈǈ�޻���'�@��������ϼԼ������f�@������)�&�)�6�B�I�O�[�h�tāĈĂā�x�t�h�O�6�)ĚĔčċčĚĦĳĵĹĳĦĚĚĚĚĚĚĚĚ�n�n�b�U�N�N�U�b�j�n�{�}ńŇŉŇ�{�{�n�n���������������
��#�&�0�9�5�0�#��
���������������	������
�	��	�	�������߽�����(�4�;�>�@�D�4�(������Y�U�[�m�{�����ʼ����#�%��
��ּʼ��Y�r�Y�@�9�I�L�R�Y�e�r�~���������������~�r�.�)�"���	���������	��!�.�0�4�;�?�;�.��
������������!��������������������������������� �'�&��������ÿùôìäàù����������������������������þìÓÇÄÀ�|�{Çàì�������������Ҽ4�-�(�-�3�4�@�B�I�E�@�5�4�4�4�4�4�4�4�4�/�#�#����#�/�<�H�U�a�m�a�`�U�L�H�<�/�Ϲù��������������¹Ϲ߹���	�����ܹ�����������������������������������������E*E E%E*E7ECEGECE;E7E*E*E*E*E*E*E*E*E*E*������������$�$�(�$� ������������������������������������ʾ׾۾оʾ������������������������������� �&�(�"��	������������(�-�(�"�����������������������ɺ���!�-�7�!�����ɺ����F�A�F�S�U�_�l�t�x�������x�l�_�S�F�F�F�F�������������������Ŀѿݿ�޿ֿҿϿϿĿ�����	����!�.�.�8�.�&�!�������S�H�G�C�G�S�`�l�y�������������y�l�`�S�S������������!�-�4�:�>�:�9�-�!����E�E�E�E�E�E�FFF$F1F=FJFKFLF;F$FE�E�E�EuEtEuExE�E�E�E�E�E�EuEuEuEuEuEuEuEuEuEu���������������������������������������� 4 M w O : 8 N � P F ^  [ q 9 C U > @ = r ~ $ = H 2 ? N 2 9 a D  M 2 � f ? c O V b 8 2 ] r D 7 D p & l H q m U 3 c ` Q 1 \ Q 7 ? � = Z ` s ? r + m 9 N  �  ]  �  �  �    S  O  :  H    �  j  �  V  �  s  �  �  �  �  �  �  �  @  �  >  O  �    h  �  $  �  �  O  �  =  �  �  3    :  T  ;  �  f  �  /  �  �  �  }  >  L  �  �  W  �  L  B  9  H  �  �  �  �  �  �  �  �  8  @    M  1�D��;o;o�t�:�o��o��9X��C��Y����
�'�`B�D������49X����'��ͽq���H�9��w��9X�ě��\)�t���9X�t�����o�49X�]/��h�H�9�+���49X�Y��D���\)�o��P�\)���#�
��-�8Q��w�'Y��@��<j��Q�ixսD���@���-�]/�����H�9��\)��1�e`B��O߽�o��O߽� Ž�O߽�Q콕��������w���-��{������vɽ��`B�EA�`ZB�{B�PB��B�WB��B��B��B�B̝B!��A���A�O�B�zB(�B4Bh�B&��B(УA�J�B4mB�<B�B*)B(��B؁A���B�B�/BA�B��BBāB��BiB��B��B�}B
WBB#&B �B�AB
��B�B� B�B)B+nB	~B]B-^B ��B��B��B0�B�B
 �B(��B
דBv�B�BEB��B��B"ucB�B@�B��B�Bq�Bp�B"�iB�VBo�B��B=CA�&B�2B�JB7�B��B=�B5�BʥB̹BAB"rA��A���BA�B?�B��B�B&�BB)1�A�}�B4��B�B 0BA�B(��B+�A��B��B�\B�BF�B>�B̭B)�B�	BA�B@B�GB
x�B#?/B��B�(B
�aB� B�B�B�B@$B	>�B9B-H�B!'B�B�VB�YB?�B
.�B(�LB
��B@ B=�B>0B�4B��B!�B?�BG�B�kB BNCB"�B"��B�sB��BE3A�M�A�A�i%A��6A9�\BɌA�Y}B4�A}��B	�KA]�jAFA��hA��A�Ư?A�A��A�+dA7��A#BL�AO%�B %gA�:�C��gA��GA��{A�K@�_�A��EA���B
p�A��XA� A��A�N�AZ4�A� A�#AA��OA'~�@�	@��B��@�S�A�sCA�HUA�b'A�
�AYD�A3�e@�� ?�S�A^%�A�)�A���A��A�#�@ξxAâ>�#	A�IOC��uB	ALF�A�z A��Y@>��@�o�Ay��AHAX+@e�OC��C��uA�J~A̋>A뚂A�Y�A���A8�GB<TAń�BCA|�B	H�A_[AD��A���A���A��?P5�B 0�A��hA7A3A$��B�JAO�A�RA�BC�ߴA���A���A��x@��A�L�A��B
K�A���A�ʣA��A�1BAV8�A�~�A���A�	�A'|@��@���B��@���A�YA��9A���A虡AX��A21A
%?�l�A]	A�`#A��3A��A�}�@�)�AÀ�>��kA�'&C���B�AI�QA�}tA�_�@D@�J�Ay�AA�AW�@h�C���C���A�8                           .             &                
   ,   "                        
   ,                                                z                  
   6      	      -      $         $         
                      	            	   
                           7      =   !   9            %      !      %                              '               !   )                        5                     7   #         %      %         !                     )                                                         =   !   -                                                      #               !   %                        %                     5   #         !                                    )                        O.3~NW�N>a�O%91N��O,��N�OUY7O5{N�ϮPj�UO�B�P��N�/�Oc��N�pO��N�OlO�OJ��O�e*N48$O?��Oow�O�N��eOm��O%�O`C�O9LZO�~N��*O{��No�N��OSDO�.O��NV��Nm~�Ne�MۖcN0�xNI�O��+N�P3Nl9�N��N�*|N�*�O�2�P:�/O��N۾VN �AO��O;"�OWp`N<i�N���N�t�N9oN"σN�J�O+`wOpˇNwGO��N� O17%N�n�N��O�O�N5	�N�]  e    �    }  1  �  �  �  �  �  �  e  E  ^  �  �  �  D  �  �  �  �  �    ]  �  5  		  P  �  +  �  @    �  �  7  �  �    �      �    �  �  a  ~  {  =  �  7  Y  @  �  "  �  *  A  �    ]  ;  �  �  �  %  �  [  8  �  �  �  �<t�<o<o;ě�;��
��o��`B��`B���49X�#�
�T�����
�e`B��t��u��h��o�C����ͼ��㼓t���t���t���t����㼴9X��1��������/���ͼ���`B��h��`B�+��h��h�����o�o�+�}���\)�t���P����P��w��w�#�
�0 Ž8Q�49X�]/�<j�]/�}�Y��]/�aG��aG��aG��q���y�#�y�#��+��\)��\)��\)������{��v����
#$,,$#
�����!")/3880/%"!!!!!!!!���������������������������������������������


���������������������������������������������)25CFC5)�������������������������������������������%BVbht�������t[@6$%��������������������GTaz��������vmaTO??G����		���������������
!
�����mnz������znmmmmmmmmm557>BNSZ[^][QNEB:655
#/01//%#

/0:<IUU[\ZUJI<40/-,/_bdn{��������{tnjgc_Tafmz���|}{zyoaTPMOT���������������������������������������� #'/<Hdkie]LH</.*#  ��������������������jno{�����}{nnijjjjjj�������������������TTamz~zzuomgaTMKNOTTOV[bht����{yth[OLIKO����������������������������������������
"#$#
��������sw|��������������{xs���������������������������������������������� ����������������������������������������������dnuz{}|znmidddddddddst���������vutssssss������������������������������������������������������������������������������

�������� %$%&)257?>95)#&&&&&&&&����������������������������������������BN[gtw����{tqg[QNCBB��$'(&"��������������!��������������������������������������������������������������)BM[ksp[B5 ��������������X[\cgt���������tg]VXnn{������{vnnnnnnnnn�������������������LOX[hltxvtopjh[SOKHL��������������������
!
��������������������xz��������������wusx������
����������������������������������������������������������������������_anz������������znc_����������������

	����������������������������%/;<DHJLLLH<4/-)&%%%FHMU^\UH<?FFFFFFFFFF��������������������ßàáàÝÚßàìù����������������ìß�0�-�#��#�0�<�I�J�I�H�<�0�0�0�0�0�0�0�0�t�i�t¦§¦ �
�����������������
��#�/�9�7�/�,�#��
�4�,�)�(� �(�4�4�5�A�G�K�E�A�4�4�4�4�4�4����ƳƧƧƛƛƧƮƳ���������������������U�L�H�A�@�E�D�H�U�V�a�e�n�p�p�n�a�Y�U�U�0�*��"�$�,�5�=�A�I�V�b�h�c�V�N�I�=�6�0�ݿۿѿǿĿ��ĿȿѿԿݿ�����������ݿ����������$�0�4�5�0�0�%�$�����	����ʾ������ʾ��.�T�y�����`�V�;�"�	�s�i�^�T�U�_�p�����������������������s������׿ֿٿݿ����5�g�u����s�5���	����������	���"�/�3�;�>�;�/�"��	�	�����������������������������������������������������������������*�$���
���� �*�6�C�O�R�O�N�C�6�/�*ŹŮŴŹż��������������������������ŹŹ���������(�4�A�M�O�V�M�E�A�4�(���������~�����������Ľֽٽڽнý�������ƵƟƚƏƏƚƧƳ�������������������Ƶ���������������ʾϾ̾ʾ������������������*�������*�6�C�K�V�]�\�O�C�>�;�6�*�;�4�/�*�'�$�"�/�;�H�T�a�e�n�o�m�a�R�H�;FFFF$F+F1F:F=F?FJFUFVF^F_FVFJF=F1F$F�g�[�Z�O�V�Z�g�s�v���������s�g�g�g�g�g�g���������������������"�$�/�-�"��	�����"�"����"�/�0�;�H�J�T�^�]�T�H�;�/�"�"�����������������������лۻܻڻȻû��������������������������	�����	��������s�g�T�M�P�Z�g�������������������������s�0�.�&�$��$�0�3�=�>�E�C�A�=�0�0�0�0�0�0ŹŭŠŚŘśšŬŭŹ������������������Ź����������������������������������������ŇŅńŇŔŠŭŰůŭŠŔŇŇŇŇŇŇŇŇÒÌÇ�z�n�`�U�U�a�n�zÇÓáìúöìàÒ�	���׾ƾ¾ʾ׾���(�.�>�C�G�;�.��	�z�t�g�g�o�o�t�{¢¨¦�g�a�g�q�s�������������s�g�g�g�g�g�g�g�g�������������
�����
����������������Ľý������Ľн׽ҽнĽĽĽĽĽĽĽĽĽļ�����{������������������������������������������ʼҼԼʼ���������������������ǈǀ�{�p�{ǀǈǔǗǚǔǌǈǈǈǈǈǈǈǈ�@�'���� �'�4�@�f��������������r�Y�@�B�>�B�E�M�O�[�h�t�|�t�s�h�[�Y�O�B�B�B�BĚĔčċčĚĦĳĵĹĳĦĚĚĚĚĚĚĚĚ�n�n�b�U�N�N�U�b�j�n�{�}ńŇŉŇ�{�{�n�n���������������
��#�$�0�4�0�/�#��
����������������	�������	�����������߽�����(�4�;�>�@�D�4�(������f�\�Y�c�n����ʼ����!�#��	���㼱��f�r�Y�@�9�I�L�R�Y�e�r�~���������������~�r�.�)�"���	���������	��!�.�0�4�;�?�;�.��
������������!����������������������������������&�$��������ÿùôìäàù������������������������ìãàÓÐÌÉÇÈÓàìùþ��������ùì�4�-�(�-�3�4�@�B�I�E�@�5�4�4�4�4�4�4�4�4�/�.�#�(�/�<�H�S�U�Y�U�H�@�<�/�/�/�/�/�/�ù������������ùϹҹܹ��������ܹϹ�����������������������������������������E*E E%E*E7ECEGECE;E7E*E*E*E*E*E*E*E*E*E*������������$�$�(�$� ������������������������������������ʾ׾۾оʾ������������������������������� �&�(�"��	������������(�-�(�"�����������������������ɺ���!�-�7�!�����ɺ����F�A�F�S�U�_�l�t�x�������x�l�_�S�F�F�F�F�������������������Ŀѿݿ�޿ֿҿϿϿĿ�����	����!�.�.�8�.�&�!�������S�H�G�C�G�S�`�l�y�������������y�l�`�S�S������������!�-�4�:�>�:�9�-�!����E�E�E�E�FFFF$F1F=FEFIF=F8F1F$FFE�E�EuEtEuExE�E�E�E�E�E�EuEuEuEuEuEuEuEuEuEu���������������������������������������� = M w O : 0 J � 6 ? ^  b f 0 C 8 > " D j N $ = H 2 8 N / 2 \ D  M % � b ? c O V b 8 2 ? T D 7 @ g & e H q m E 3 : ` > % \ Q 7 ? � = Z ` s ? r + [ 9 N  u  ]  �  �  �  r  !  O  9  �    �  z    �  �  '  �  @  �  �  d  �  �  @  �  �  O  �  �    �  �  �  �  O  �  =  �  �  3    :  T  _  �  f  �    Q  �  �  }  >  L  G  �  �  �  �    9  H  �  �  �  �  �  �  �  �  8  @  D  M  1  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  Y  a  d  a  Y  N  A  /    	    �  �  �  �  �  �  s  T  3    �  �  �  �  �  �  �  u  `  I  1    �  �  �  �  l  A    �  �  �  �  �  �  �  �  �  �  �  �  z  U    �  n  :    �        �  �  �  �  �  �  a  <    �  �  �  t  7  �  �  U  }  |  |  {  {  x  n  c  Y  O  A  0        �   �   �   �   �   �  �  �    !  ,  1  /  %      �  �  �  �  L    �  �  L    ^  }  �  �  �  �  �  �    N    �  z  �    �  �  u  q  D  �  �  �  �  �  �  |  i  S  9    �  �  �  �  �  `  .    �  �  �    S  j  t  x  u  r  l  k  }  �  �  �  <  �  _  �  v  �  �  �  �  �  �  �  �  n  W  9    �  �  �  \    �  �  �  �  �  �  �  |  ^  ;    �  �  �  �  �  �  L  	  �  �  -   �  �  �  �  �  �  �  �  �  }  ]  9    �  �  �  �  �  T     �  �    F  \  d  e  `  S  7    �  �  d  B  D  -  �  �    &     ,  8  C  C  A  ?  8  /  &        �  �  �  �  z  S  ,  -  ?  P  ^  Z  O  D  M  C  $  �  �  �  Z    �  �    �  �  �  �  �  �  �  �  �  �  �    o  `  P  A  1  !       �   �  h  s  �  �  �  �  �  �  �  �  �  �  �  �  z  E    �  e    �  �  �  �  �  l  S  7    �  �  �  �  �  k  P  2    �  �    g  �  �    *  8  @  D  @  -    �  p    �  N  �  g  �  �  �  �  �  �  �  �  �  �  �  �  �  `  /  �  �  z  C      �  �  �  �  �  �  �  �  X  3    �  �  |  J  
  �  U  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  ^  D  &     �   �  �  �  �  �  �  �  �  �  �  �  �  {  e  T  L  D  @  O  ^  n  �  �  �  �  �  �  �  {  o  _  K  /    �  �  �  ]  '  �  �    �  �  �  �  �  �  `  <    �  �  �  c    �  W    �  o  ]  Z  X  U  R  P  M  G  @  8  1  )  !       �   �   �   �   �  �  �  �  �  �  �  ~  m  Y  B  (  	  �  �  r  8  �  �  y  :  5  +  "        �  �  �  �  �  �  x  O    �  �  5   �   �  �  �  	  	  	  �  �  �  �  }  L    �  C  �  G  �  �  D  �  )  =  C  K  P  P  K  C  8  '    �  �  �  �  O    �  g    �  �  �  �  �  �  w  ]  f  m  I  "    �  �  /  %  �  �  #  +        �  �  �  �  �  �  �  x  n  e  [  I  3       �  �  �  �  �  �  �  �  �  �  �  }  j  V  ?    �  �  0  �  ,  @  >  <  :  7  4  0  -  )  $        	    �  �  �  �  {  �  �    �  �  �  �  �  �  �  �  x  `  H  0    
  �  �  �  �  n  �  �  ^  5  	  �  #  '    �  �  �  P    �  �  �  R  t  x  }  �  �  s  [  ?    �  �  �  �  a  /  �  �  |    �  7  0  ,  *  )    	  �  �  �  �  �  �  �  `  $  �  �  '  �  �  �  {  s  l  e  ^  N  9  $    �  �  �  �  ~  [  0     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    "  /  6  ,  #    �  �  �  b  -  �  �  k  *  �  �  `    �  �  �  �  �  �  �  �  �  �    }  z  �  �  �  �  �  �              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    o  ^  M  :  &    �  �  �  �  d  3    �  �  N     �   �  
�  o  .  z  �  �  �  }  W  %  �  [  
�  	�  �    '  �  �  B  �  �  �  �  �  �  �  �  �  �  �  �  Y  ,  �  �  �  K  >  7  �  |  v  p  k  e  a  \  W  S  U  ^  g  p  x  U  #  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    s  h  \  O  A  3  %  ^  a  ]  M  9  #    �  �  �  �  E  �  �  m  %  �  �  1  u  <  `  {  }  }  z  p  c  I  *    �  �  y  I    �  �  �  �  {  q  f  Z  N  @  2  $      �  �  �  �  �  �  �  �  �  �  &  <  0    �  �  �  Y    �  ~    �  !  �  �  ?  �      �  �  �  �  n  R  @  (    �  �  �  �  �  u  O  !  �  �  j  7  *        �  �  �  �  �  �  �  �  j  H  
  �  w      �  Y  O  E  :  0  #    �  �  �  �  �  i  G  $     �  �  �  p  �  @  <  7  6  <  ;  $  �  �  �  8  �  T  �  5    �  o  O  �  �  �  x  k  a  [  [  ]  _  _  [  T  J  0    �  o     �  �  ?  �  �  �    !       �  �  �  q  -  �  �  "  �  �    �  �  �  �  �  w  c  P  <  )       �   �   �   �   �   �   �   o  �  �  �  �    (  !    �  �  �  c  #  �  �  &  �  K  �  [  �  �    (  1  7  A  A  <  4  %  0  +    �  �  �  S  .  �  �  �  �  �  �  �  �  �  {  r  i  `  W  Y  q  �  �  �  �  �    �  �  �  �  �  �  u  \  >    �  �  9  �  �  W  	  �  h  ]  +  �  �  |  :  �  �  z  E    �  �  �  ~  [  9    �  �  ;  #        �  �  �  �  �  m  A    �  �  }  a  ?  "    �  �  �  U  3    9  W  �  a  +  �  �  Y  �  �  1  u  t  =  �  s  \  E  -    �  �  �  �  u  S  .    �  �  �  O    �  �  �  i  H  (     �  �  ~  P  !  �  �    :  �  �  V    e  %  "    �  �  �  �  j  B    �  �  �  R  !  �  �  �  E    �  �  �  �  �  �  w  Q  (  �  �  �  s  K    �  �  `    �  [  Q  H  <  /  !      �  �  �  �  �  �  �  v  K  -  &    8  +        �  �  �  �  �  �  �  �  �  �  {  r  |  �  �  �  �  �  �  �  �  �  �  �  �  p  F    �  �  �  �  o  \  D  g  f  d  �  l  3  �  �  �  J  �  �  G  �  �  /  �  Q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  Y  �  p  D  %    �  �  �  �  k  K  ,    �  �  4  �  �  *  �